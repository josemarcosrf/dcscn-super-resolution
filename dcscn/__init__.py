import torch
import numpy as np
from skimage.measure import compare_psnr, compare_ssim


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    def __getattr__(self, attr):
        return self.get(attr)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self


def to_numpy(t):
    """
    Torch Tensor Variable to numpy array
    Args:
        t (torch.autograd.Variable): Variable tensor to convert
    Returns:
        numpy array with t data
    """
    return t.data.cpu().numpy()


def to_tensor(ndarray):
    """Converts a np.array into pytorch.tensor

    Args:
        ndarray (np.array): numpy array to convert to tensor
    """
    return torch.from_numpy(ndarray)


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
