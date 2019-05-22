import torch
import numpy as np
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim


def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t


def bicubic_upsampling(image_arr, scale_factor):
    """Upsamples a given image in array format by a given scale factor
    using bicubic interpolation.
    """
    h, w = image_arr.shape[:2]
    new_h = h * scale_factor
    new_w = w * scale_factor
    image = Image.fromarray(image_arr, "RGB")
    image = image.resize([new_w, new_h], resample=Image.BICUBIC)
    return np.asarray(image)


def quad_to_image(quad):
    """Transforms a quad-image (a tensor of shape BxCxHxW) into
    a tensor of shape 1xC/2*HxC/2*W

    Arguments:
        quad Tensor -- Tensor of shape CxHxW
    """
    b, n_channels, h, w = quad.data.shape
    up_factor = n_channels // 2

    # Tensor with the same device and dtype
    placeholder = quad.new_zeros((b, 1, up_factor * h, up_factor * w))

    for i in range(n_channels):
        x = int(i // 2)
        y = int(i % 2)
        placeholder[:, 0, y*h:(y+1)*h, x*w:(x+1)*w] = quad[:, i, :, :]

    return placeholder


def compute_psnr_and_ssim(image1, image2, border_size=0):
    """Computes the
        'Peak Signal to Noise Ratio' and the 'Structural Similarity Index'

       This function has been taken from:
       https://github.com/jiny2001/dcscn-super-resolution/blob/ver1/helper/utilty.py

       And adapted to work with the code in this repo:

    """
    # TODO: review image dimension ordering!
    # add channel information
    if len(image1.shape) == 2:
        image1 = image1.reshape(image1.shape[0], image1.shape[1], 1)
    if len(image2.shape) == 2:
        image2 = image2.reshape(image2.shape[0], image2.shape[1], 1)

    # ensure same image size
    if image1.shape[0] != image2.shape[0] or \
        image1.shape[1] != image2.shape[1] or \
            image1.shape[2] != image2.shape[2]:
        return None

    # cast to double for metric comparison
    image1 = image1.astype(np.double)
    image2 = image2.astype(np.double)

    # trim borders if applicable
    if border_size > 0:
        image1 = image1[border_size:-border_size, border_size:-border_size, :]
        image2 = image2[border_size:-border_size, border_size:-border_size, :]

    # compute difference metrics
    psnr = compare_psnr(image1, image2, data_range=255)
    ssim = compare_ssim(image1, image2,
                        win_size=11,
                        gaussian_weights=True,
                        multichannel=True,
                        K1=0.01, K2=0.03,
                        sigma=1.5, data_range=255)
    return psnr, ssim
