""" Python modules for Lanczos interpolation. """


import torch
import numpy as np


def lanczos_kernel(dx, a=3, N=None, dtype=None, device=None):
    '''
    Generates 1D Lanczos kernels for translation and interpolation.
    Args:
        dx : float, tensor (batch_size, 1), the translation in pixels to shift an image.
        a : int, number of lobes in the kernel support.
            If N is None, then the width is the kernel support (length of all lobes),
            S = 2(a + ceil(dx)) + 1.
        N : int, width of the kernel.
            If smaller than S then N is set to S.
    Returns:
        k: tensor (?, ?), lanczos kernel
    '''

    if not torch.is_tensor(dx):
        dx = torch.tensor(dx, dtype=dtype, device=device)

    if device is None:
        device = dx.device

    if dtype is None:
        dtype = dx.dtype

    D = dx.abs().ceil().int()
    S = 2 * (a + D) + 1  # width of kernel support

    S_max = S.max() if hasattr(S, 'shape') else S

    if (N is None) or (N < S_max):
        N = S

    Z = (N - S) // 2  # width of zeros beyond kernel support

    start = (-(a + D + Z)).min()
    end = (a + D + Z + 1).max()
    x = torch.arange(start, end, dtype=dtype, device=device).view(1, -1) - dx
    px = (np.pi * x) + 1e-3

    sin_px = torch.sin(px)
    sin_pxa = torch.sin(px / a)

    k = a * sin_px * sin_pxa / px**2  # sinc(x) masked by sinc(x/a)

    return k


def lanczos_shift(img, shift, p=3, a=3):
    '''
    Shifts an image by convolving it with a Lanczos kernel.
    Lanczos interpolation is an approximation to ideal sinc interpolation,
    by windowing a sinc kernel with another sinc function extending up to a
    few nunber of its lobes (typically a=3).

    Args:
        img : tensor (batch_size, channels, height, width), the images to be shifted
        shift : tensor (batch_size, 2) of translation parameters (dy, dx)
        p : int, padding width prior to convolution (default=3)
        a : int, number of lobes in the Lanczos interpolation kernel (default=3)
    Returns:
        I_s: tensor (batch_size, channels, height, width), shifted images
    '''

    dtype = img.dtype

    if len(img.shape) == 2:
        img = img[None, None].repeat(1, shift.shape[0], 1, 1)  # batch of one image
    elif len(img.shape) == 3:  # one image per shift
        assert img.shape[0] == shift.shape[0]
        img = img[None, ]

    # Apply padding

    padder = torch.nn.ReflectionPad2d(p)  # reflect pre-padding
    I_padded = padder(img)

    # Create 1D shifting kernels

    y_shift = shift[:, [0]]
    x_shift = shift[:, [1]]

    k_y = (lanczos_kernel(y_shift, a=a, N=None, dtype=dtype)
           .flip(1)  # flip axis of convolution
           )[:, None, :, None]  # expand dims to get shape (batch, channels, y_kernel, 1)
    k_x = (lanczos_kernel(x_shift, a=a, N=None, dtype=dtype)
           .flip(1)
           )[:, None, None, :]  # shape (batch, channels, 1, x_kernel)

    # Apply kernels

    I_s = torch.conv1d(I_padded,
                       groups=k_y.shape[0],
                       weight=k_y,
                       padding=[k_y.shape[2] // 2, 0])  # same padding
    I_s = torch.conv1d(I_s,
                       groups=k_x.shape[0],
                       weight=k_x,
                       padding=[0, k_x.shape[3] // 2])

    I_s = I_s[..., p:-p, p:-p]  # remove padding

    return I_s.squeeze()  # , k.squeeze()
