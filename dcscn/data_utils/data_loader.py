import os
import argparse
import logging
import coloredlogs

import numpy as np


from torchvision import transforms
from torchvision.datasets import ImageFolder

from dcscn.data_utils import (plot_images,
                              convert_rgb_to_ycbcr,
                              convert_to_grayscale)


logger = logging.getLogger(__name__)


# TODO: Convert images to 1 channel Y from YCbCr or grascale


class DataLoader():

    def __init__(self, data_dir, augment_level=4):
        self.data_dir = data_dir
        self.augment_level = augment_level
        self.augmentations = None
        self._build_augmentations()

    def load_transform(self):
        """Loads all images from a directory containing subdirectories with image files
        and applies all tranformations rules for the given augmentation level.

        Returns:
            Dataset -- List of (List of PIL images, class label)
        """
        try:

            sets = os.listdir(self.data_dir)
            logger.info("Found the following datasets: {}".format(sets))
            logger.info("Building {}-level augmented data.".format(self.augment_level))

            # load data and apply transformations
            dataset = ImageFolder(
                root=self.data_dir,
                transform=lambda x: self.apply_transform(x))

            # In this case we don't care about the label
            total_imgs = sum(len(imgs) for imgs, _ in dataset)
            logger.info("Total images after augmentation: {}".format(total_imgs))

            return dataset

        except Exception as e:
            logger.error("Error while applying tranformations")
            logger.exception(e)

    def apply_transform(self, img, to_tensor=True):
        """Applies a set of transformations on a given PIL image."""
        img = convert_rgb_to_ycbcr(img)
        if to_tensor:
            return [transforms.Compose([
                augmentation_trf,
                transforms.ToTensor()
                ])(img)[0, :, :] for augmentation_trf in self.augmentations]
        return [taugmentation_trf(img)
                for taugmentation_trf in self.augmentations]

    def _extract_patches(self, img_tensor, size=32, step=16):
        """Given an tensor representing a 1-channel image
        extracts patches of size 'size' with steps of size 'step'.

        Arguments:
            img_tensor {Tensor} -- Representing an image with only 1 channel

        Returns:
            Tensor -- of shape B x size x size
        """
        return img_tensor[0, :, :].unfold(0, size=size, step=step) \
            .unfold(1, size=size, step=step) \
            .reshape(-1, size, size)

    def _build_augmentations(self):
        if self.augment_level < 2:
            raise ValueError("Augmentation level must be at least 2.")

        r1 = (90, 90)       # right rotation range
        r2 = (270, 270)     # left rotation range

        # array of torchvision transformations
        self.augmentations = [
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
        ][:self.augment_level]


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
    loader = DataLoader(args.data_dir, args.augment_level)
    image_dataset = loader.load_transform()

    # plot a random sample of the images and its transformations
    indices = np.random.choice(len(image_dataset), args.n_samples)
    for i in indices:
        plot_images(image_dataset[i][0])
        # imgs_tensor = [transforms.ToTensor()(img) for img in image_dataset[i][0]]
        # patches = [extract_patches(img_t) for img_t in imgs_tensor]
        # plot_images([transforms.ToPILImage()(p[0, :, :]) for p in patches])
