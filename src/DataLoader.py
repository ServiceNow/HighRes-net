""" Python script to load, augment and preprocess batches of data """

from collections import OrderedDict
import numpy as np
from os.path import join, exists, basename, isfile

import glob
import skimage
from skimage import io

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def get_patch(img, x, y, size=32):
    """
    Slices out a square patch from `img` starting from the (x,y) top-left corner.
    If `im` is a 3D array of shape (l, n, m), then the same (x,y) is broadcasted across the first dimension,
    and the output has shape (l, size, size).
    Args:
        img: numpy.ndarray (n, m), input image
        x, y: int, top-left corner of the patch
        size: int, patch size
    Returns:
        patch: numpy.ndarray (size, size)
    """
    
    patch = img[..., x:(x + size), y:(y + size)]   # using ellipsis to slice arbitrary ndarrays
    return patch


class ImageSet(OrderedDict):
    """
    An OrderedDict derived class to group the assets of an imageset, with a pretty-print functionality.
    """

    def __init__(self, *args, **kwargs):
        super(ImageSet, self).__init__(*args, **kwargs)

    def __repr__(self):
        dict_info = f"{'name':>10} : {self['name']}"
        for name, v in self.items():
            if hasattr(v, 'shape'):
                dict_info += f"\n{name:>10} : {v.shape} {v.__class__.__name__} ({v.dtype})"
            else:
                dict_info += f"\n{name:>10} : {v.__class__.__name__} ({v})"
        return dict_info


def sample_clearest(clearances, n=None, beta=50, seed=None):
    """
    Given a set of clearances, samples `n` indices with probability proportional to their clearance.
    Args:
        clearances: numpy.ndarray, clearance scores
        n: int, number of low-res views to read
        beta: float, inverse temperature. beta 0 = uniform sampling. beta +infinity = argmax.
        seed: int, random seed
    Returns:
        i_sample: numpy.ndarray (n), sampled indices
    """
    
    if seed is not None:
        np.random.seed(seed)
        
    e_c = np.exp(beta * clearances / clearances.max()) ##### FIXME: This is numerically unstable. 
    p = e_c / e_c.sum()
    idx = range(len(p))
    i_sample = np.random.choice(idx, size=n, p=p, replace=False)
    return i_sample


def read_imageset(imset_dir, create_patches=False, patch_size=64, seed=None, top_k=None, beta=0.):
    """
    Retrieves all assets from the given directory.
    Args:
        imset_dir: str, imageset directory.
        create_patches: bool, samples a random patch or returns full image (default).
        patch_size: int, size of low-res patch.
        top_k: int, number of low-res views to read.
            If top_k = None (default), low-views are loaded in the order of clearance.
            Otherwise, top_k views are sampled with probability proportional to their clearance.
        beta: float, parameter for random sampling of a reference proportional to its clearance.
        load_lr_maps: bool, reads the status maps for the LR views (default=True).
    Returns:
        dict, collection of the following assets:
          - name: str, imageset name.
          - lr: numpy.ndarray, low-res images.
          - hr: high-res image.
          - hr_map: high-res status map.
          - clearances: precalculated average clearance (see save_clearance.py)
    """

    # Read asset names
    idx_names = np.array([basename(path)[2:-4] for path in glob.glob(join(imset_dir, 'QM*.png'))])
    idx_names = np.sort(idx_names)
    
    clearances = np.zeros(len(idx_names))
    if isfile(join(imset_dir, 'clearance.npy')):
        try:
            clearances = np.load(join(imset_dir, 'clearance.npy'))  # load clearance scores
        except Exception as e:
            print("please call the save_clearance.py before call DataLoader")
            print(e)
    else:
        raise Exception("please call the save_clearance.py before call DataLoader")

    if top_k is not None and top_k > 0:
        top_k = min(top_k, len(idx_names))
        i_samples = sample_clearest(clearances, n=top_k, beta=beta, seed=seed)
        idx_names = idx_names[i_samples]
        clearances = clearances[i_samples]
    else:
        i_clear_sorted = np.argsort(clearances)[::-1]  # max to min
        clearances = clearances[i_clear_sorted]
        idx_names = idx_names[i_clear_sorted]

    lr_images = np.array([io.imread(join(imset_dir, f'LR{i}.png')) for i in idx_names], dtype=np.uint16)

    hr_map = np.array(io.imread(join(imset_dir, 'SM.png')), dtype=np.bool)
    if exists(join(imset_dir, 'HR.png')):
        hr = np.array(io.imread(join(imset_dir, 'HR.png')), dtype=np.uint16)
    else:
        hr = None  # no high-res image in test data

    if create_patches:
        if seed is not None:
            np.random.seed(seed)

        max_x = lr_images[0].shape[0] - patch_size
        max_y = lr_images[0].shape[1] - patch_size
        x = np.random.randint(low=0, high=max_x)
        y = np.random.randint(low=0, high=max_y)
        lr_images = get_patch(lr_images, x, y, patch_size)  # broadcasting slicing coordinates across all images
        hr_map = get_patch(hr_map, x * 3, y * 3, patch_size * 3)

        if hr is not None:
            hr = get_patch(hr, x * 3, y * 3, patch_size * 3)

    # Organise all assets into an ImageSet (OrderedDict)
    imageset = ImageSet(name=basename(imset_dir),
                        lr=np.array(lr_images),
                        hr=hr,
                        hr_map=hr_map,
                        clearances=clearances,
                        )

    return imageset




class ImagesetDataset(Dataset):
    """ Derived Dataset class for loading many imagesets from a list of directories."""

    def __init__(self, imset_dir, config, seed=None, top_k=-1, beta=0.):

        super().__init__()
        self.imset_dir = imset_dir
        self.name_to_dir = {basename(im_dir): im_dir for im_dir in imset_dir}
        self.create_patches = config["create_patches"]
        self.patch_size = config["patch_size"]
        self.seed = seed  # seed for random patches
        self.top_k = top_k
        self.beta = beta
        
    def __len__(self):
        return len(self.imset_dir)        

    def __getitem__(self, index):
        """ Returns an ImageSet dict of all assets in the directory of the given index."""    

        if isinstance(index, int):
            imset_dir = [self.imset_dir[index]]
        elif isinstance(index, str):
            imset_dir = [self.name_to_dir[index]]
        elif isinstance(index, slice):
            imset_dir = self.imset_dir[index]
        else:
            raise KeyError('index must be int, string, or slice')

        imset = [read_imageset(imset_dir=dir_,
                                  create_patches=self.create_patches,
                                  patch_size=self.patch_size,
                                  seed=self.seed,
                                  top_k=self.top_k,
                                  beta=self.beta,)
                    for dir_ in tqdm(imset_dir, disable=(len(imset_dir) < 11))]

        if len(imset) == 1:
            imset = imset[0]

        imset_list = imset if isinstance(imset, list) else [imset]
        for i, imset_ in enumerate(imset_list):
            imset_['lr'] = torch.from_numpy(skimage.img_as_float(imset_['lr']).astype(np.float32))
            if imset_['hr'] is not None:
                imset_['hr'] = torch.from_numpy(skimage.img_as_float(imset_['hr']).astype(np.float32))
                imset_['hr_map'] = torch.from_numpy(imset_['hr_map'].astype(np.float32))
            imset_list[i] = imset_

        if len(imset_list) == 1:
            imset = imset_list[0]

        return imset
