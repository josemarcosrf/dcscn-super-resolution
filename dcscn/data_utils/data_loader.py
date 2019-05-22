import os
import argparse
import logging
import coloredlogs

import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import ImageFolder


logger = logging.getLogger(__name__)


# TODO: Convert images to 1 channel Y from YCbCr or grascale


def plot_images(images):
    n = len(images)
    plt.figure()
    for i, img in enumerate(images):
        image = np.array(img)
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(image, interpolation='none')
        ax.set_title('Transform #{}'.format(i))
        ax.axis('off')

    plt.show()


def convert_rgb_to_ycbcr(pil_img):
    """Converts a given PIL image from RGB to YCbCr
    """
    return pil_img.convert('YCbCr')


def convert_to_grayscale(pil_img):
    return pil_img.convert('LA')


def build_transformations(augment_level):
    if augment_level < 2:
        raise ValueError("Augmentation level must be at least 2!")

    r1 = (90, 90)       # right rotation
    r2 = (270, 270)     # left rotation

    # transformation names to save the images
    names = ["_v", "_h", "_vh", "_r1", "_r2", "_r1v", "_r2v"]

    # array of torchvision transformations
    transformations = [
        # none transformation
        lambda x: x,
        # vertical and horizontal flip
        transforms.RandomVerticalFlip(1),
        transforms.RandomHorizontalFlip(1),
        # compistion of H + V flips
        transforms.Compose([
            transforms.RandomVerticalFlip(1),
            transforms.RandomHorizontalFlip(1),
        ]),
        # rotations
        transforms.RandomRotation(r1),
        transforms.RandomRotation(r2),
        # rotation + vertical flips
        transforms.Compose([transforms.RandomRotation(r1),
                            transforms.RandomVerticalFlip(1)]),
        transforms.Compose([transforms.RandomRotation(r2),
                            transforms.RandomVerticalFlip(1)]),
    ]

    return names[:augment_level], transformations[:augment_level]


def apply_transform(transformations, img, to_tensor=False):
    """applies a set of transformations on a given PIL image
    """
    if to_tensor:
        return [transforms.Compose([t, transforms.ToTensor()])(img)
                for t in transformations]
    return [t(img)
            for t in transformations]


def load_transform(directory, augment_level):
    try:

        sets = os.listdir(directory)
        logger.info("Found the following datasets: {}".format(sets))

        logger.info("Building {}-level augmented data.".format(augment_level))
        trf_names, transformations = build_transformations(augment_level)

        # load data and apply transformations
        train_data = ImageFolder(
            root=directory,
            transform=lambda x: apply_transform(transformations, x)
        )

        # In this case we don't care about the label
        total_imgs = sum(len(imgs) for imgs, _ in train_data)
        logger.info("Total images after augmentation: {}".format(total_imgs))

        return train_data

    except Exception as e:
        logger.error("Error while applying tranformations")
        logger.exception(e)


def extract_patches(img_tensor, size=32, step=16):
    logger.debug("img_tensor shape: {}".format(img_tensor.shape))
    return img_tensor[0, :, :].unfold(0, size=size, step=step) \
        .unfold(1, size=size, step=step) \
        .reshape(-1, size, size)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",
                        help="Data directory")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Number fo samples to plot")
    parser.add_argument("--augment_level", type=int, default=4,
                        help=("Augmentation level:\n",
                              ">=2: vertical flip"
                              ">=3: horizontal flip"
                              ">=4: vertical+horizontal flip"
                              ">=5: 90 degrees rotation"
                              ">=6: -90 degrees rotation"
                              ">=7: 90 degrees rotation + vertical flip"
                              ">=8: -90 degrees rotation + vertical flip"
                              ))
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # configure the logger
    coloredlogs.install(logger=logger, level=logging.DEBUG,
                        format="%(filename)s:%(lineno)s - %(message)s")

    # load and transform from the original data
    image_dataset = load_transform(args.data_dir, args.augment_level)

    # plot a random sample of the images and its transformations
    indices = np.random.choice(len(image_dataset), args.n_samples)
    for i in indices:
        plot_images(image_dataset[i][0])
        # imgs_tensor = [transforms.ToTensor()(img) for img in image_dataset[i][0]]
        # patches = [extract_patches(img_t) for img_t in imgs_tensor]
        # plot_images([transforms.ToPILImage()(p[0, :, :]) for p in patches])
