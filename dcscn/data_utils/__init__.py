import math
import random
import numpy as np
import matplotlib.pyplot as plt

from itertools import islice
from PIL import Image


def plot_images(images):
    n = len(images)
    rows = int(math.sqrt(n))
    cols = int(math.ceil(n/rows))
    plt.figure()
    for i, img in enumerate(images):
        image = np.array(img)
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(image, interpolation='none')
        ax.set_title('Transform #{}'.format(i))
        ax.axis('off')

    plt.show()


def bicubic_interpolation(image_arr, scale_factor):
    """Intepolates a given image in numpy array format (C x H x W)
    by a given scale factor using bicubic interpolation.
    """
    _, h, w = image_arr.shape
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    image = Image.fromarray(image_arr.swapaxes(0, 2), "RGB")
    image = image.resize([new_w, new_h], resample=Image.BICUBIC)
    return np.asarray(image, dtype=np.float32).swapaxes(0, 2)


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


def convert_rgb_to_ycbcr(pil_img):
    """ Converts a given PIL image to YCbCr """
    return pil_img.convert('YCbCr')


def convert_to_grayscale(pil_img):
    """ Converts a PIL image to grayscale. """
    return pil_img.convert('LA')


def add_channel_dim(img_arr):
    """Given a numpy array of shape N x H x W adds a channel
    dimension: N x C x H x W.

    Args:
        img_arr (np.array): numpy 4-dimensional array

    Returns:
        np.array: numpy 4-dimensional array
    """
    return img_arr[:, None, :, :]


def chunk(iterable, c_size, stack_func=np.stack):
    """
    Given an iterable yields chunks of size 'c_size'.
    The iterable can be an interator, we do not assume iterable to have 'len' method.
    Args:
        iterable (iterable): to be partitioned in chunks
        c_size (int): size of the chunks to be produced
    Returns:
        (generator) of elements of size 'c_size' from the given iterable
    """
    it = iter(iterable)
    while True:
        chunk = list(islice(it, c_size))
        if not chunk:
            return
        yield stack_func(chunk)


def parallel_shuffle(*args):
    """
    Shuffle n lists concurrently.

    Args:
        *args: list of iterables to shuffle concurrently

    Returns:
        shuffled iterables
    """
    combined = list(zip(*args))
    random.shuffle(combined)
    args = zip(*combined)
    return [list(x) for x in args]


def parallel_split(split_ratio, *args):
    """
    Splits n lists concurrently

    Args:
        *args: list of iterables to split
        split_ratio (float): proportion to split the lists into two parts
    Returns:

    """
    all_outputs = []
    for a in args:
        split_idx = int(len(a) * split_ratio)
        all_outputs.append((a[:split_idx], a[split_idx:]))
    return all_outputs

