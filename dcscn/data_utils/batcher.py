import logging
import coloredlogs
import numpy as np

from dcscn.data_utils.data_loader import DataLoader
from dcscn.data_utils import (chunk,
                              add_channel_dim,
                              bicubic_interpolation)

logger = logging.getLogger(__name__)


class DataBatcher():

    """Class to yield training and testing batches for the trainer.
    Makes use of the DataLoader to ingest all images and apply relevant
    augmentation and other transformations.
    Once the dataset is loaded creates image patches.
    """

    def __init__(self, train_dir, test_dir,
                 scale_factor=2, patch_size=64, stride=32):
        self.stride = stride
        self.patch_size = patch_size
        self.scale_factor = scale_factor

        # initialize the dataloader for the train set
        training_dataset = DataLoader(train_dir).load_transform()

        logger.info("Extracting training patches")
        self.training_patches = self._get_image_patches(training_dataset)
        logger.info("Extracted a total of {} "
                    "training patches "
                    "with shape {}".format(self.training_patches.shape[0],
                                           self.training_patches.shape[1:]))

        # # initialize the dataloader for the test set
        # test_dataset = DataLoader(test_dir).load_transform()

        # logger.info("Extracting testing patches")
        # self.testing_patches = self._get_image_patches(test_dataset)
        # logger.info("Extracted a total of {} "
        #             "testing patches "
        #             "with shape {}".format(self.testing_patches.shape[0],
        #                                    self.testing_patches.shape[1:]))

    def _get_image_patches(self, dataset):
        """Generate all patches and create a numpy tensor of
        shape N x 1 x H x W
        Then we shuffle along the first axis (along the patches)

        Returns:
            np.array: containing image patches (N x 1 x H x W)
        """
        # concatenta all patches along the batch dimension
        patches = np.concatenate([
            add_channel_dim(self._extract_patches(img_tensor))
            for images, _ in dataset
            for img_tensor in images], 0)
        # shuffle the batch
        np.random.shuffle(patches)
        return patches

    def _extract_patches(self, img_tensor):
        """Given an tensor representing a 1-channel image (H x W)
        extracts patches of size 'size' with steps of size 'step'.

        Arguments:
            img_tensor {Tensor} -- Representing an image with only 1 channel

        Returns:
            Tensor -- of shape B x size x size
        """
        return img_tensor.unfold(0, size=self.patch_size, step=self.stride) \
            .unfold(1, size=self.patch_size, step=self.stride) \
            .reshape(-1, self.patch_size, self.patch_size)

    def get_training_batches(self, batch_size):
        """Returns an iterator yielding a chunk (batch) of tuples containing
        inut and target.

        Args:
            batch_size (int): size of each batch
        """
        # A training sample is a downsampled image as X
        # and the orginal patch as Y
        s_factor = 1.0 / self.scale_factor
        inputs = np.concatenate([
            add_channel_dim(bicubic_interpolation(x, s_factor))
            for x in self.training_patches
        ])

        logger.debug("Inputs are of shape: {}".format(inputs.shape))
        logger.debug("Ouputs are of shape: {}".format(self.training_patches.shape))

        return zip(
            chunk(inputs, batch_size),
            chunk(self.training_patches, batch_size)
        )

    def get_val_batch(self, batch_size):
        pass


if __name__ == "__main__":
    # configure the logger
    coloredlogs.install(logger=logger, level=logging.DEBUG,
                        format="%(filename)s:%(lineno)s - %(message)s")

    batcher = DataBatcher("data/train", "data/eval")

    for batch in batcher.get_training_batches(10):
        inp, out = batch
        logger.debug(inp.shape)
        logger.debug(out.shape)
        exit()

