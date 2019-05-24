import logging
import coloredlogs
import torch

from dcscn import dotdict
from dcscn.net import DCSCN
from dcscn.trainer import Trainer
from dcscn.data_utils.batcher import DataBatcher


logger = logging.getLogger(__name__)


# For now hardocded
default_config = {
    'num_epochs': 100,
    'initial_lr': 0.002,
    'lr': 0.002,
    'batch_size': 64,
    'use_cuda': True,
    'train_data_path': './data/mini/train',
    'eval_data_path': './data/mini/eval',
    'tf_log_dir': './logs',
    'checkpoint_path': './checkpoints',
    'eval_every': 10,
    'patience': 5,
    'save_name': 'default'
}


if __name__ == "__main__":

    # configure the logger
    coloredlogs.install(logger=logger,
                        level=logging.DEBUG,
                        format="%(filename)s:%(lineno)s - %(message)s")

    conf = dotdict(default_config)

    # build the model (default model parameters)
    logger.info("Building DCSCN model")
    model = DCSCN()
    if conf.use_cuda and torch.cuda.is_available():
        model.cuda()

    # configure the batching
    logger.info("Building data batcher")
    batcher = DataBatcher(conf.train_data_path,
                          conf.eval_data_path)

    # configure the training (default training parameters)
    logger.info("Building trainer and starting training")
    trainer = Trainer(model, batcher, conf)
    trainer.train()
