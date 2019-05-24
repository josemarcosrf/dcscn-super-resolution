import logging
import coloredlogs
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from dcscn import to_numpy
from dcscn.model_utils.tf_logger import Logger
from dcscn.model_utils import save_model

logger = logging.getLogger(__name__)

# configure the logger
coloredlogs.install(logger=logger,
                    level=logging.DEBUG,
                    format="%(filename)s:%(lineno)s - %(message)s")

# this breaks the logging misserably....
# from tensorboardX import SummaryWriter
# writer = SummaryWriter()


class Trainer:

    def __init__(self, model, batcher, train_cfg):
        """Trainer helper class.
        Given a pytorch.nn.module and a data 'Batcher' train the model using
        the 'train_cfg' configuration.
        The 'model' must expose 'train_batch' and 'eval_batch' functions.
        The 'batcher' must expose 'get_training_batches' and 'get_test_batches' functions.

        Args:
            model (nn.Module): model to be trained
            batcher (batcher): Batcher object yielding training and evaluation batches
            train_cfg (dotdict): Containing the training paremeters:
                {
                    'use_cuda': False,
                    'tf_log_dir': './logs',
                    'num_epochs': 10,
                    'batch_size': 16,
                    'checkpoint_path': './checkpoints',
                    'patience': 5,
                    'save_name': 'default'
                }
        """
        logger.info("Model received:")
        try:
            model.print_summary()
        except expression as identifier:
            logger.info(model)

        self.model = model
        self.batcher = batcher

        self.train_cfg = train_cfg

        if train_cfg.use_cuda:
            logger.info("Moving model to CUDA device")
            self.model.cuda()

        # this must be called after moving the model to CPU or GPU
        self.model.make_optimizer(lr=train_cfg.lr)

        # Set the TF logger
        self.tf_logger = Logger(train_cfg.tf_log_dir)

    def train(self, control_metric):
        """
        Trains the given model with the batches provided by the batcher

        Returns:
            trained model
        """

        epochs_it = tqdm(range(self.train_cfg.num_epochs))
        epoch_loss = float('inf')
        train_loss = 0
        epoch_loss = 0
        best_measure = 0
        ctrl_measures = []
        for epoch in epochs_it:
            self.model.train()
            epochs_it.set_description("epoch {} - "
                                      "loss: {:.3f}".format(epoch, epoch_loss))

            # iterate over all batches
            for b, train_batch in enumerate(tqdm(
                self.batcher.get_training_batches(self.train_cfg.batch_size)
            )):
                batch_x, batch_y = train_batch
                epoch_loss += self.model.train_batch(
                    batch_x,
                    batch_y,
                    use_cuda=self.train_cfg.use_cuda
                )
            epoch_loss /= (b + 1)

            # eval
            # TODO: make the model agnostic at eval_batch metrics / results
            if epoch % self.train_cfg.eval_every == 0:
                self.model.eval()
                val_metrics = defaultdict(int)

                for b, val_batch in enumerate(tqdm(
                        self.batcher.get_val_batch(self.train_cfg.batch_size)
                )):
                    batch_x, batch_y = val_batch
                    res = self.model.eval_batch(
                        batch_x,
                        batch_y,
                        use_cuda=self.train_cfg.use_cuda
                    )
                    for k, v in res.items():
                        val_metrics[k] += v

                msgs = []
                for k, v in val_metrics.items():
                    val_metrics[k] = v / (b + 1)
                    msgs.append("{}: {:.3f}".format(k, val_metrics[k]))
                msg = " | ".join(msgs)
                tqdm.write("Epoch {} ==> {}".format(epoch, msg))

                if val_metrics[control_metric] > best_measure:
                    best_measure = val_metrics[control_metric]
                    # save the model
                    checkpoint_name = "{}_epoch={}_{}={:.3f}".format(
                        self.train_cfg.save_name, epoch,
                        control_metric, best_measure
                    )
                    tqdm.write('Saving model as: {}'.format(checkpoint_name))
                    save_model(self.model,
                               self.train_cfg.checkpoint_path,
                               checkpoint_name)

                # Basic early stopping on validation accuracy
                # TODO: configurable to track different metrics
                if self.train_cfg.patience:
                    ctrl_measures.append(val_metrics[control_metric])
                    ctrl_measures = ctrl_measures[-self.train_cfg.patience:]
                    if len(ctrl_measures) >= self.train_cfg.patience:
                        if all([a > b
                                for a, b in zip(ctrl_measures[:-1],
                                                ctrl_measures[1:])]):
                            logger.warning(
                                "Early stopping due to accuracy decrease"
                                " for the last {} evaluations"
                                "".format(self.train_cfg.patience))
                            break
                        elif all([np.isclose(a, b, atol=1e-4)
                                  for a, b in zip(ctrl_measures[:-1],
                                                  ctrl_measures[1:])]):
                            logger.warning(
                                "Early stopping due to accuracy stagnation"
                                " for the last {} evaluations"
                                "".format(self.train_cfg.patience)
                            )
                            break

            # ============ TensorBoard logging ============
            # scalar values
            info = {
                'train_loss': epoch_loss
            }
            info.update(val_metrics)
            for tag, value in info.items():
                self.tf_logger.scalar_summary(tag, value, epoch + 1)

            # Log values and gradients of the parameters (histogram)
            for tag, value in self.model.named_parameters():
                try:
                    tag = tag.replace('.', '/')
                    self.tf_logger.histo_summary(tag, to_numpy(value), epoch + 1)
                    self.tf_logger.histo_summary(tag + '/grad', to_numpy(value.grad), epoch + 1)
                except Exception as e:
                    logger.exception(e)
                    logger.error("tag {}".format(tag))

            train_loss += epoch_loss

        train_loss /= self.train_cfg.num_epochs
        return {
            'val_loss': val_metrics['val_loss'],
            'val_acc': val_metrics['val_acc'],
            'train_loss': train_loss
        }
