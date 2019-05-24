import coloredlogs
import logging
import torch
import numpy as np

from torch import nn
from torch import optim
from torch.nn import functional as F

from dcscn import to_numpy, compute_psnr_and_ssim
from dcscn.data_utils import quad_to_image


logger = logging.getLogger(__name__)

# configure the logger
coloredlogs.install(logger=logger, level=logging.INFO,
                    format="%(filename)s:%(lineno)s - %(message)s")


# TODO: Add CNN weights penalty to the loss function
# TODO: Add initialization to all CNN meights before PReLU:
# https://pytorch.org/docs/master/nn.html#torch.nn.init.kaiming_uniform_
# PReLu and bias to 0
# CNN with torch.nn.init.kaiming_uniform_


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        logger.debug("Initializing {}".format(m))
        nn.init.kaiming_uniform_(m.weight, a=0)
        nn.init.zeros_(m.bias)


class DCSCN(nn.Module):

    """
        Fast and Accurate Image Super Resolution by Deep CNN
        with Skip Connection and Network in Network
        https://arxiv.org/pdf/1707.05425.pdf
    """

    default_params = {
        'scale_factor': 2,
        'dropout': 0.8,
        # 'conv_n_filters': [96, 76, 65, 55, 47, 39, 32],
        'conv_n_filters': [96, 81, 70, 60, 50, 41, 32],
        'conv_kernels': [(3, 3), (3, 3), (3, 3),
                         (3, 3), (3, 3), (3, 3), (3, 3)],
        'reconstruction_n_filters': {
            'A1': 64,
            'B1': 32,
            'B2': 32
        },
        'reconstruction_kernels': {
            'A1': (1, 1),
            'B1': (1, 1),
            'B2': (3, 3)
        }
    }

    def __init__(self, params={}):
        super().__init__()

        self._set_parameters(params)
        self._build_model()
        self.apply(weights_init)

    def _set_parameters(self, params):
        params = self.default_params
        params.update(params)

        # training parameters
        self.dropout = params['dropout']
        # feature extraction parameters
        self.conv_n_filters = params['conv_n_filters']
        self.conv_kernels = params['conv_kernels']
        # reconstruction parameters
        self.scale_factor = params['scale_factor']
        self.reconstruction_n_filters = params['reconstruction_n_filters']
        self.reconstruction_kernels = params['reconstruction_kernels']

    def forward(self, x):
        """Forward model pass.

        Arguments:
            x Tensor -- Batch of image tensors o shape: B x C x H x W

        Returns:
            Tensor -- Super Resolution upsampled image
        """
        # bi-cubic upsampling
        x_up = F.interpolate(x, mode='bicubic',
                             scale_factor=self.scale_factor)

        # convolutions and skip connections
        features = []
        for conv in self.conv_sets:
            x_new = conv.forward(x)
            features.append(x_new)
            x = x_new
            logger.debug("features: {}".format(x_new.data.shape))

        # concatenation 1: through filter dimensions
        x = torch.cat(features, 1)
        logger.debug("Concatenation - "
                     "before reconstruction: {}".format(x.data.shape))

        # reconstruction part
        a1_out = self.reconstruction['A1'].forward(x)
        logger.debug("A1: {}".format(a1_out.data.shape))

        b1_out = self.reconstruction['B1'].forward(x)
        logger.debug("B1: {}".format(b1_out.data.shape))

        b2_out = self.reconstruction['B2'].forward(b1_out)
        logger.debug("B2: {}".format(b2_out.data.shape))

        # concatenation 2 & last convolution
        x = torch.cat([a1_out, b2_out], 1)
        x = self.l_conv.forward(x)      # outputs a quad-image

        # network output + bicubic upsampling:
        # the 4 channels of the output represent the 4 corners
        # of the resulting image
        x = quad_to_image(x) + x_up

        return x

    def _build_model(self):
        # feature extraction network
        in_channels = 1
        self.conv_sets = nn.ModuleList()
        for s_i, n_filters in enumerate(self.conv_n_filters):
            self.conv_sets.append(
                self._build_conv_set(in_channels, n_filters,
                                     kernel=self.conv_kernels[s_i]))
            in_channels = n_filters

        # reconstruction network
        in_channels = np.sum(self.conv_n_filters)
        self.reconstruction = nn.ModuleDict()
        # A1
        self.reconstruction['A1'] = self._build_reconstruction_conv(
            in_channels,
            self.reconstruction_n_filters['A1'],
            kernel=self.reconstruction_kernels['A1']
        )
        # B1
        self.reconstruction['B1'] = self._build_reconstruction_conv(
            in_channels,
            self.reconstruction_n_filters['B1'],
            kernel=self.reconstruction_kernels['B1']
        )
        # B2
        self.reconstruction['B2'] = self._build_reconstruction_conv(
            self.reconstruction_n_filters['B1'],
            self.reconstruction_n_filters['B2'],
            kernel=self.reconstruction_kernels['B2'],
            padding=1
        )
        # last convolution
        inp_channels = self.reconstruction_n_filters['B2'] + \
            self.reconstruction_n_filters['A1']
        self.l_conv = nn.Conv2d(
            inp_channels,
            self.scale_factor**2,
            (1, 1)
        )

    def train_batch(self, batch_x, batch_y, use_cuda=False):

        logger.debug("Received x ({})"
                     " with shape: {}".format(batch_x.dtype, batch_x.shape))
        logger.debug("Received y ({})"
                     " with shape: {}".format(batch_y.dtype, batch_y.shape))

        # Reset gradients
        self.zero_grad()
        # forward pass
        if use_cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        output = self.forward(batch_x)
        loss = F.mse_loss(output, batch_y)
        # backward pass
        loss.backward()
        # optimization step
        self.optimizer.step()

        return loss.data.cpu()

    def eval_batch(self, batch_x, batch_y, use_cuda=False):
        logger.debug("Received x ({})"
                     " with shape: {}".format(batch_x.dtype, batch_x.shape))
        logger.debug("Received y ({})"
                     " with shape: {}".format(batch_y.dtype, batch_y.shape))

        metrics = {}

        if use_cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        output = self.forward(batch_x)
        loss = F.mse_loss(output, batch_y)

        # compute evaluation metrics
        metrics['mse'] = loss.data.cpu()
        metrics['psnr'], metrics['ssim'] = self._eval_metrics(output, batch_y)
        return metrics

    def make_optimizer(self, lr):
        self.optimizer = optim.Adam(params=self.parameters(), lr=lr)

    def print_summary(self):
        from torchsummary import summary
        summary(self, (1, 64, 64))

    def _eval_metrics(self, outputs, targets):
        # rescale to 255
        # TODO: Review proper rescaling
        outputs *= 255
        targets *= 255

        # convert to numpy an remove channel dimension
        total_psnr = total_ssim = 0
        for out, y in zip(to_numpy(outputs), to_numpy(targets)):
            psnr, ssim = compute_psnr_and_ssim(out[0, :, :], y[0, :, :])
            total_psnr += psnr
            total_ssim += ssim

        n = outputs.shape[0]
        return total_psnr / n, total_ssim / n

    def _build_conv_set(self, in_channels, out_channels, kernel):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                       stride=1, padding=1,
                                       kernel_size=kernel),
                             nn.PReLU(),
                             nn.Dropout(p=self.dropout))

    def _build_reconstruction_conv(self, in_channels,
                                   out_channels, kernel, padding=0):
        return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels,
                                                padding=padding,
                                                kernel_size=kernel),
                             nn.PReLU(),
                             nn.Dropout(p=self.dropout))


if __name__ == "__main__":

    import numpy as np

    model = DCSCN({})
    model.print_summary()

    if torch.cuda.is_available():
        model.cuda()

    # try some random input
    x = torch.FloatTensor(np.random.rand(1, 1, 32, 32))
    x = x.cuda()
    print(x.is_cuda)
    print(model.forward(x).data.shape)


