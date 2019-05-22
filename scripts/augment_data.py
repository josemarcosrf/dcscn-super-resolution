import os
import argparse
import logging
import coloredlogs

import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import ImageFolder


logger = logging.getLogger(__name__)


def plot_trasnformations(images):
    n = len(images)
    plt.figure()
    for i, img in enumerate(images):
        image = np.array(img)
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(image, interpolation='none')
        ax.set_title('Transform #{}'.format(i))
        ax.axis('off')

    plt.show()


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


def apply_transform(transformations, img):
    return [t(img) for t in transformations]


def transform_all(directory, augment_level):
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

        logger.debug("train_data type: {}".format(type(train_data)))

        # create a dictionary of images indexed by dataset
        for images, c in train_data:
            plot_trasnformations(images)
            print(images[0].size)
            print(np.array(images[0]).shape)
            exit()

    except Exception as e:
        logger.error("Error while applying tranformations")
        logger.exception(e)
        exit




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Data directory")
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

    transform_all(args.data_dir, args.augment_level)
