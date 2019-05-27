import logging
import coloredlogs
import torch

from dcscn import (dotdict, to_numpy,
                   compute_psnr_and_ssim)

from dcscn.net import DCSCN
from dcscn.training.trainer import Trainer
from dcscn.training.checkpointer import Checkpointer
from dcscn.training.metrics import MetricTracker
from dcscn.data_utils.batcher import DataBatcher


logger = logging.getLogger(__name__)


# For now hardocded
default_config = dotdict({
    'num_epochs': 100,
    'eval_every': 5,
    'lr': 0.002,
    'batch_size': 64,
    'use_cuda': True,
    'train_data_path': './data/train',
    'eval_data_path': './data/eval',
    'tf_log_dir': './logs',
    'checkpoint_path': './checkpoints',
})


def create_trainer_helpers():
    # MSE stagnation tracker to decrease learning rate
    mse_stagnation_tracker = MetricTracker(
        'mse', MetricTracker.is_stagnated, {'patience': 5, 'tolerance': 1e-4}
    )

    # Checkpointer: save model only on SPNR improvement
    psnr_tracker = MetricTracker('psnr', MetricTracker.is_max, {})
    checkpointer = Checkpointer(
        default_config.checkpoint_path,
        "fulldata-model_epoch={epoch}_mse={mse}_psnr={psnr}",
        metric_tracker=psnr_tracker
    )

    return mse_stagnation_tracker, checkpointer


if __name__ == "__main__":

    # configure the logger
    coloredlogs.install(logger=logger,
                        level=logging.DEBUG,
                        format="%(filename)s:%(lineno)s - %(message)s")

    conf = dotdict(default_config)

    # build the model (default model parameters)
    logger.info("Building DCSCN model")
    model = DCSCN()
    if default_config.use_cuda and torch.cuda.is_available():
        model.cuda()

    # configure the batching
    logger.info("Building data batcher")
    batcher = DataBatcher(default_config.train_data_path,
                          default_config.eval_data_path,
                          default_config.batch_size)

    # configure the training (default training parameters)
    logger.info("Building training helpers (MSE stagnation tracker)")
    mse_stagnation_tracker, checkpointer = create_trainer_helpers()

    logger.info("Building trainer and starting training")
    trainer = Trainer(model, batcher,
                      lr=default_config.lr,
                      num_epochs=default_config.num_epochs,
                      eval_every=default_config.eval_every,
                      use_cuda=default_config.use_cuda,
                      tf_log_dir=default_config.tf_log_dir,
                      checkpointer=checkpointer,
                      lr_updater=mse_stagnation_tracker)
    trainer.train()
