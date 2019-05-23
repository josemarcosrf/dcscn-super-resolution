import math
import numpy as np
import matplotlib.pyplot as plt


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


def convert_rgb_to_ycbcr(pil_img):
    """ Converts a given PIL image to YCbCr """
    return pil_img.convert('YCbCr')


def convert_to_grayscale(pil_img):
    """ Converts a PIL image to grayscale. """
    return pil_img.convert('LA')
