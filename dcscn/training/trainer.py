import logging
import coloredlogs
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from dcscn import to_numpy
from dcscn.training import save_model

logger = logging.getLogger(__name__)

# configure the logger
coloredlogs.install(logger=logger,
                    level=logging.DEBUG,
                    format="%(filename)s:%(lineno)s - %(message)s")

# this breaks the logging misserably....
# from tensorboardX import SummaryWriter
# writer = SummaryWriter()


# TODO: Accept event listeners (e.g.: MSE stagnates -> lr /= 2)
# TODO: Let saving the model be handled by the checkpointer


class Trainer:

    # Known actions that can be taken during training
    # Expects a mapping from this actions to a trigger criteria:
    # (Criteria: MSE increases -> Action: early stop)

    def __init__(self, model, batcher, checkpointer,
                 num_epochs=100, eval_every=10, lr=1e-3,
                 use_cuda=True, tf_log_dir=None,
                 early_stop_checker=None, lr_updater=None):
        """Trainer helper class.
        Given a pytorch.nn.module and a data 'Batcher' train the model:
        The 'model' must expose 'train_batch' and 'eval_batch' functions.
        The 'batcher' must expose 'get_training_batches' and 'get_test_batches'
        functions.

        Args:
            model (nn.Module): model to be trained
            batcher (batcher): Batcher object yielding training
                               and evaluation batches.
        """
        logger.info("Model received:")
        try:
            model.print_summary()
        except Exception as e:
            logger.warning("Error calling model 'print_summary' "
                           "function: {}".format(e))
            logger.info(model)

        self.model = model
        self.batcher = batcher

        # TODO: fixed number of Events for now
        self.lr_updater = lr_updater
        self.early_stop_checker = early_stop_checker
        self.checkpointer = checkpointer

        # set initial training parameters
        self.num_epochs = num_epochs
        self.lr = lr
        self.use_cuda = use_cuda
        self.tf_log_dir = tf_log_dir
        self.eval_every = eval_every

        if self.use_cuda:
            logger.info("Moving model to CUDA device")
            self.model.cuda()

        # this must be called after moving the model to CPU or GPU
        self.model.make_optimizer(lr=self.lr)

        # Set the TF logger
        self.tf_logger = None
        if self.tf_log_dir:
            from dcscn.training.tf_logger import Logger
            self.tf_logger = Logger(self.tf_log_dir)

    def train(self):
        """
        Trains the given model with the batches provided by the batcher.

        Returns:
            trained model
        """
        tracking_metrics = defaultdict(list)

        self.epochs_it = tqdm(range(self.num_epochs))
        for epoch in self.epochs_it:
            # train the entire epoch
            self._train_epoch(epoch)

            # evaluate on the entire evaluation set and save model
            if epoch % self.eval_every == 0:
                val_metrics = self._eval(epoch)
                for k, v in val_metrics.items():
                    tracking_metrics[k].append(v)

            # TensorBoard logging
            self._log(val_metrics, epoch)

            # ================ Training Checks - hardcoded for now ============
            # checkpoint model
            if self.checkpointer.check(epoch=epoch,
                                       metrics=tracking_metrics):
                ckpt_name = self.checkpointer.format_ckpt_name(
                    epoch=epoch, metrics=tracking_metrics
                )
                self._save_model(ckpt_name)

            # check if early stopping
            if self.early_stop_checker and \
                    self.early_stop_checker.check(tracking_metrics):
                logger.warning("Early stopping criteria met. Stopping!")
                break

            # check learning-rate updater
            if self.lr_updater and \
                    self.lr_updater.check(tracking_metrics):
                # TODO: Accept a function to update
                self.lr /= 2
                tqdm.write("Reducing LR to {}".format(self.lr))
                self.model.make_optimizer(lr=self.lr)
            # ================ Training Checks ================================

        return dict([(k, v[-1])
                     for k, v in tracking_metrics.items()])

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        # iterate over all batches
        for b, train_batch in enumerate(
                tqdm(self.batcher.get_training_batches())):
            batch_x, batch_y = train_batch
            epoch_loss += self.model.train_batch(
                batch_x,
                batch_y,
                use_cuda=self.use_cuda
            )
        epoch_loss /= (b + 1)
        self.epochs_it.set_description(
            "epoch {} - loss: {:.3f}".format(epoch, epoch_loss))
        return epoch_loss

    def _eval(self, epoch):
        """Evaluate the model on the entire evaluation set."""
        self.model.eval()
        val_metrics = defaultdict(int)

        for b, val_batch in enumerate(tqdm(self.batcher.get_val_batch())):
            batch_x, batch_y = val_batch
            res = self.model.eval_batch(
                batch_x,
                batch_y,
                use_cuda=self.use_cuda
            )
            for k, v in res.items():
                val_metrics[k] += v

        msgs = []
        for k, v in val_metrics.items():
            val_metrics[k] = v / (b + 1)
            msgs.append("{}: {:.3f}".format(k, val_metrics[k]))

        msg = " | ".join(msgs)
        tqdm.write("Epoch {} ==> {}".format(epoch, msg))
        return val_metrics

    def _save_model(self, checkpoint_name):
        tqdm.write('Saving model as: {}'.format(checkpoint_name))
        save_model(self.model,
                   self.checkpointer.checkpoint_path,
                   checkpoint_name)

    def _log(self, metrics, epoch):
        if self.tf_logger:
            # scalar values
            for tag, value in metrics.items():
                self.tf_logger.scalar_summary(tag, value, epoch + 1)

            # Log values and gradients of the parameters (histogram)
            for tag, value in self.model.named_parameters():
                try:
                    tag = tag.replace('.', '/')
                    self.tf_logger.histo_summary(tag,
                                                 to_numpy(value),
                                                 epoch + 1)
                    self.tf_logger.histo_summary(tag + '/grad',
                                                 to_numpy(value.grad),
                                                 epoch + 1)
                except Exception as e:
                    logger.exception(e)
                    logger.error("tag {}".format(tag))
