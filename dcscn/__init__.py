import numpy as np
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim


def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t


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
