""" Python script to train HRNet + shiftNet for multi frame super resolution (MFSR) """

import json
import os
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.optim as optim
import argparse
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from DeepNetworks.HRNet import HRNet
from DeepNetworks.ShiftNet import ShiftNet

from DataLoader import ImagesetDataset
from Evaluator import shift_cPSNR
from utils import getImageSetDirectories, readBaselineCPSNR, collateFunction
from tensorboardX import SummaryWriter


def register_batch(shiftNet, lrs, reference):
    """
    Registers images against references.
    Args:
        shiftNet: torch.model
        lrs: tensor (batch size, views, W, H), images to shift
        reference: tensor (batch size, W, H), reference images to shift
    Returns:
        thetas: tensor (batch size, views, 2)
    """
    
    n_views = lrs.size(1)
    thetas = []
    for i in range(n_views):
        theta = shiftNet(torch.cat([reference, lrs[:, i : i + 1]], 1))
        thetas.append(theta)
    thetas = torch.stack(thetas, 1)

    return thetas


def apply_shifts(shiftNet, images, thetas, device):
    """
    Applies sub-pixel translations to images with Lanczos interpolation.
    Args:
        shiftNet: torch.model
        images: tensor (batch size, views, W, H), images to shift
        thetas: tensor (batch size, views, 2), translation params
    Returns:
        new_images: tensor (batch size, views, W, H), warped images
    """
    
    batch_size, n_views, height, width = images.shape
    images = images.view(-1, 1, height, width)
    thetas = thetas.view(-1, 2)
    new_images = shiftNet.transform(thetas, images, device=device)

    return new_images.view(-1, n_views, images.size(2), images.size(3))


def get_loss(srs, hrs, hr_maps, metric='cMSE'):
    """
    Computes ESA loss for each instance in a batch.
    Args:
        srs: tensor (B, W, H), super resolved images
        hrs: tensor (B, W, H), high-res images
        hr_maps: tensor (B, W, H), high-res status maps
    Returns:
        loss: tensor (B), metric for each super resolved image.
    """
    
    # ESA Loss: https://kelvins.esa.int/proba-v-super-resolution/scoring/
    criterion = nn.MSELoss(reduction='none')
    if metric == 'masked_MSE':
        loss = criterion(hr_maps * srs, hr_maps * hrs)
        return torch.mean(loss, dim=(1, 2))
    nclear = torch.sum(hr_maps, dim=(1, 2))  # Number of clear pixels in target image
    bright = torch.sum(hr_maps * (hrs - srs), dim=(1, 2)).clone().detach() / nclear  # Correct for brightness
    loss = torch.sum(hr_maps * criterion(srs + bright.view(-1, 1, 1), hrs), dim=(1, 2)) / nclear  # cMSE(A,B) for each point
    if metric == 'cMSE':
        return loss
    return -10 * torch.log10(loss)  # cPSNR


def get_crop_mask(patch_size, crop_size):
    """
    Computes a mask to crop borders.
    Args:
        patch_size: int, size of patches
        crop_size: int, size to crop (border)
    Returns:
        torch_mask: tensor (1, 1, 3*patch_size, 3*patch_size), mask
    """
    
    mask = np.ones((1, 1, 3 * patch_size, 3 * patch_size))  # crop_mask for loss (B, C, W, H)
    mask[0, 0, :crop_size, :] = 0
    mask[0, 0, -crop_size:, :] = 0
    mask[0, 0, :, :crop_size] = 0
    mask[0, 0, :, -crop_size:] = 0
    torch_mask = torch.from_numpy(mask).type(torch.FloatTensor)
    return torch_mask


def trainAndGetBestModel(fusion_model, regis_model, optimizer, dataloaders, baseline_cpsnrs, config):
    """
    Trains HRNet and ShiftNet for Multi-Frame Super Resolution (MFSR), and saves best model.
    Args:
        fusion_model: torch.model, HRNet
        regis_model: torch.model, ShiftNet
        optimizer: torch.optim, optimizer to minimize loss
        dataloaders: dict, wraps train and validation dataloaders
        baseline_cpsnrs: dict, ESA baseline scores
        config: dict, configuration file
    """
    np.random.seed(123)  # seed all RNGs for reproducibility
    torch.manual_seed(123)

    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]
    n_views = config["training"]["n_views"]
    min_L = config["training"]["min_L"]  # minimum number of views
    beta = config["training"]["beta"]

    subfolder_pattern = 'batch_{}_views_{}_min_{}_beta_{}_time_{}'.format(
        batch_size, n_views, min_L, beta, f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S-%f}")

    checkpoint_dir_run = os.path.join(config["paths"]["checkpoint_dir"], subfolder_pattern)
    os.makedirs(checkpoint_dir_run, exist_ok=True)

    tb_logging_dir = config['paths']['tb_log_file_dir']
    logging_dir = os.path.join(tb_logging_dir, subfolder_pattern)
    os.makedirs(logging_dir, exist_ok=True)

    writer = SummaryWriter(logging_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_score = 100

    P = config["training"]["patch_size"]
    offset = (3 * config["training"]["patch_size"] - 128) // 2
    C = config["training"]["crop"]
    torch_mask = get_crop_mask(patch_size=P, crop_size=C)
    torch_mask = torch_mask.to(device)  # crop borders (loss)

    fusion_model.to(device)
    regis_model.to(device)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['training']['lr_decay'],
                                               verbose=True, patience=config['training']['lr_step'])

    for epoch in tqdm(range(1, num_epochs + 1)):

        # Train
        fusion_model.train()
        regis_model.train()
        train_loss = 0.0  # monitor train loss

        # Iterate over data.
        for lrs, alphas, hrs, hr_maps, names in tqdm(dataloaders['train']):

            optimizer.zero_grad()  # zero the parameter gradients
            lrs = lrs.float().to(device)
            alphas = alphas.float().to(device)
            hr_maps = hr_maps.float().to(device)
            hrs = hrs.float().to(device)

            # torch.autograd.set_detect_anomaly(mode=True)
            srs = fusion_model(lrs, alphas)  # fuse multi frames (B, 1, 3*W, 3*H)

            # Register batch wrt HR
            shifts = register_batch(regis_model,
                                    srs[:, :, offset:(offset + 128), offset:(offset + 128)],
                                    reference=hrs[:, offset:(offset + 128), offset:(offset + 128)].view(-1, 1, 128, 128))
            srs_shifted = apply_shifts(regis_model, srs, shifts, device)[:, 0]

            # Training loss
            cropped_mask = torch_mask[0] * hr_maps  # Compute current mask (Batch size, W, H)
            # srs_shifted = torch.clamp(srs_shifted, min=0.0, max=1.0)  # correct over/under-shoots
            loss = -get_loss(srs_shifted, hrs, cropped_mask, metric='cPSNR')
            loss = torch.mean(loss)
            loss += config["training"]["lambda"] * torch.mean(shifts)**2

            # Backprop
            loss.backward()
            optimizer.step()
            epoch_loss = loss.detach().cpu().numpy() * len(hrs) / len(dataloaders['train'].dataset)
            train_loss += epoch_loss

        # Eval
        fusion_model.eval()
        val_score = 0.0  # monitor val score

        for lrs, alphas, hrs, hr_maps, names in dataloaders['val']:
            lrs = lrs.float().to(device)
            alphas = alphas.float().to(device)
            hrs = hrs.numpy()
            hr_maps = hr_maps.numpy()

            srs = fusion_model(lrs, alphas)[:, 0]  # fuse multi frames (B, 1, 3*W, 3*H)

            # compute ESA score
            srs = srs.detach().cpu().numpy()
            for i in range(srs.shape[0]):  # batch size

                if baseline_cpsnrs is None:
                    val_score -= shift_cPSNR(np.clip(srs[i], 0, 1), hrs[i], hr_maps[i])
                else:
                    ESA = baseline_cpsnrs[names[i]]
                    val_score += ESA / shift_cPSNR(np.clip(srs[i], 0, 1), hrs[i], hr_maps[i])

        val_score /= len(dataloaders['val'].dataset)

        if best_score > val_score:
            torch.save(fusion_model.state_dict(),
                       os.path.join(checkpoint_dir_run, 'HRNet.pth'))
            torch.save(regis_model.state_dict(),
                       os.path.join(checkpoint_dir_run, 'ShiftNet.pth'))
            best_score = val_score

        writer.add_image('SR Image', (srs[0] - np.min(srs[0])) / np.max(srs[0]), epoch, dataformats='HW')
        error_map = hrs[0] - srs[0]
        writer.add_image('Error Map', error_map, epoch, dataformats='HW')
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/val_loss", val_score, epoch)
        scheduler.step(val_score)
    writer.close()


def main(config):
    """
    Given a configuration, trains HRNet and ShiftNet for Multi-Frame Super Resolution (MFSR), and saves best model.
    Args:
        config: dict, configuration file
    """
    
    # Reproducibility options
    np.random.seed(0)  # RNG seeds
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize the network based on the network configuration
    fusion_model = HRNet(config["network"])
    regis_model = ShiftNet()

    optimizer = optim.Adam(list(fusion_model.parameters()) + list(regis_model.parameters()), lr=config["training"]["lr"])  # optim
    # ESA dataset
    data_directory = config["paths"]["prefix"]

    baseline_cpsnrs = None
    if os.path.exists(os.path.join(data_directory, "norm.csv")):
        baseline_cpsnrs = readBaselineCPSNR(os.path.join(data_directory, "norm.csv"))

    train_set_directories = getImageSetDirectories(os.path.join(data_directory, "train"))

    val_proportion = config['training']['val_proportion']
    train_list, val_list = train_test_split(train_set_directories,
                                            test_size=val_proportion,
                                            random_state=1, shuffle=True)

    # Dataloaders
    batch_size = config["training"]["batch_size"]
    n_workers = config["training"]["n_workers"]
    n_views = config["training"]["n_views"]
    min_L = config["training"]["min_L"]  # minimum number of views
    beta = config["training"]["beta"]

    train_dataset = ImagesetDataset(imset_dir=train_list, config=config["training"],
                                    top_k=n_views, beta=beta)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=n_workers,
                                  collate_fn=collateFunction(min_L=min_L),
                                  pin_memory=True)

    val_dataset = ImagesetDataset(imset_dir=val_list, config=config["training"],
                                  top_k=n_views, beta=beta)
    val_dataloader = DataLoader(val_dataset, batch_size=1,
                                shuffle=False, num_workers=n_workers,
                                collate_fn=collateFunction(min_L=min_L),
                                pin_memory=True)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    # Train model
    torch.cuda.empty_cache()

    trainAndGetBestModel(fusion_model, regis_model, optimizer, dataloaders, baseline_cpsnrs, config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path of the config file", default='config/config.json')

    args = parser.parse_args()
    assert os.path.isfile(args.config)

    with open(args.config, "r") as read_file:
        config = json.load(read_file)

    main(config)
