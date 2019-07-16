""" Python utilities """

import csv
import numpy as np
import os
import warnings

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from skimage import transform, img_as_float, exposure
import torch


def readBaselineCPSNR(path):
    """
    Reads the baseline cPSNR scores from `path`.
    Args:
        filePath: str, path/filename of the baseline cPSNR scores
    Returns:
        scores: dict, of {'imagexxx' (str): score (float)}
    """
    scores = dict()
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            scores[row[0].strip()] = float(row[1].strip())
    return scores


def getImageSetDirectories(data_dir):
    """
    Returns a list of paths to directories, one for every imageset in `data_dir`.
    Args:
        data_dir: str, path/dir of the dataset
    Returns:
        imageset_dirs: list of str, imageset directories
    """
    
    imageset_dirs = []
    for channel_dir in ['RED', 'NIR']:
        path = os.path.join(data_dir, channel_dir)
        for imageset_name in os.listdir(path):
            imageset_dirs.append(os.path.join(path, imageset_name))
    return imageset_dirs


    
class collateFunction():
    """ Util class to create padded batches of data. """

    def __init__(self, min_L=32):
        """
        Args:
            min_L: int, pad length
        """
        
        self.min_L = min_L

    def __call__(self, batch):
        return self.collateFunction(batch)

    def collateFunction(self, batch):
        """
        Custom collate function to adjust a variable number of low-res images.
        Args:
            batch: list of imageset
        Returns:
            padded_lr_batch: tensor (B, min_L, W, H), low resolution images
            alpha_batch: tensor (B, min_L), low resolution indicator (0 if padded view, 1 otherwise)
            hr_batch: tensor (B, W, H), high resolution images
            hm_batch: tensor (B, W, H), high resolution status maps
            isn_batch: list of imageset names
        """
        
        lr_batch = []  # batch of low-resolution views
        alpha_batch = []  # batch of indicators (0 if padded view, 1 if genuine view)
        hr_batch = []  # batch of high-resolution views
        hm_batch = []  # batch of high-resolution status maps
        isn_batch = []  # batch of site names

        train_batch = True

        for imageset in batch:

            lrs = imageset['lr']
            L, H, W = lrs.shape

            if L >= self.min_L:  # pad input to top_k
                lr_batch.append(lrs[:self.min_L])
                alpha_batch.append(torch.ones(self.min_L))
            else:
                pad = torch.zeros(self.min_L - L, H, W)
                lr_batch.append(torch.cat([lrs, pad], dim=0))
                alpha_batch.append(torch.cat([torch.ones(L), torch.zeros(self.min_L - L)], dim=0))

            hr = imageset['hr']
            if train_batch and hr is not None:
                hr_batch.append(hr)
            else:
                train_batch = False

            hm_batch.append(imageset['hr_map'])
            isn_batch.append(imageset['name'])

        padded_lr_batch = torch.stack(lr_batch, dim=0)
        alpha_batch = torch.stack(alpha_batch, dim=0)

        if train_batch:
            hr_batch = torch.stack(hr_batch, dim=0)
            hm_batch = torch.stack(hm_batch, dim=0)

        return padded_lr_batch, alpha_batch, hr_batch, hm_batch, isn_batch


def imsetshow(imageset, k=None, show_map=True, show_histogram=True, figsize=None, **kwargs):
    """
    # TODO flake8 W605 invalid escape sequence '\m'
    Shows the imageset collection of high-res and low-res images with clearance maps.
    Args:
        k : int, number of low-res views to show. Default option (k=0) shows all.
        show_map : bool (default=True), shows a row of subplots with a mask under each image.
        show_histogram : bool (default=True), shows a row of subplots with a color histogram
                         under each image.
        figsize : tuple (default=None), overrides the figsize. If None, a default size is used.

        **kwargs : arguments passed to `plt.imshow`.
    """
    
    lr = imageset['lr']
    hr = imageset['hr']
    hr_map = imageset['hr_map']
    i_ref = 0
    n_lr = k if k is not None else lr.shape[0]
    has_hr = True if hr is not None else False
    n_rows = 1 + show_map + show_histogram

    fig = plt.figure(figsize=(3 * (n_lr + has_hr), 3 * n_rows) if figsize is None else figsize)
    sns.set_style('white')
    plt.set_cmap('viridis')

    lr_ma = np.array(lr).ravel()
    min_v, max_v = lr_ma.min(), lr_ma.max()
    col_start = 0

    if has_hr:

        min_v, max_v = min(min_v, hr.min()), max(max_v, hr.max())
        ax = fig.add_subplot(n_rows, n_lr + 1, 1, xticks=[], yticks=[])
        im = ax.imshow(hr, **kwargs)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax.set_title('HR')

        if show_map:
            ax = fig.add_subplot(n_rows, n_lr + 1, n_lr + 2, xticks=[], yticks=[])
            ax.imshow(hr_map, **kwargs)
            numel = hr_map.shape[0] * hr_map.shape[1]
            ax.set_title(f'HR status map ({100 * hr_map.sum() / numel:.0f}%)')

        if show_histogram:
            ax = fig.add_subplot(n_rows, n_lr + 1, (n_rows - 1) * (n_lr + 1) + 1, yticks=[])
            hist, hist_centers = exposure.histogram(np.array(hr), nbins=65536)
            ax.plot(hist_centers, hist, lw=2)
            ax.set_title('color histogram')
            ax.legend(['$\mu = ${:.2f}\n$\sigma = ${:.2f}'.format(hr.mean(), hr.std())], loc='upper right')

        col_start += 1

    for i in range(n_lr):

        ax = fig.add_subplot(n_rows, n_lr + 1 if has_hr else n_lr,
                             col_start + i + 1, xticks=[], yticks=[])
        im = ax.imshow(lr[i], filternorm=False, **kwargs)  # low-res
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax.set_title(f'LR-{i}' + ' (reference)' * (i == i_ref))

        if show_histogram:
            ax = fig.add_subplot(n_rows, n_lr + 1 if has_hr else n_lr,
                                 (n_rows - 1) * (n_lr + 1) + col_start + i + 1, yticks=[])
            hist, hist_centers = exposure.histogram(np.array(lr[i]), nbins=65536)
            ax.plot(hist_centers, hist, lw=2)
            ax.set_xlim(min_v, max_v)
            ax.legend(['$\mu = ${:.2f}\n$\sigma = ${:.2f}'.format(lr[i].mean(), lr[i].std())],
                      loc='upper right')

    fig.tight_layout()
