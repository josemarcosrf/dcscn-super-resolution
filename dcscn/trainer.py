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

#        # save the model
# self._save_model(epoch, control_metric, best_measure)


class MetricTracker():
    def __init__(self, metric_name, checking_func, tracking_criteria):
        self.metric_name = metric_name
        self.checking_func = checking_func
        self.best_measure = None
        self.tracking_criteria = tracking_criteria

    def check(self, val_metrics):
        measurements = val_metrics[self.metric_name]
        return self.checking_func(measurements,
                                  **self.tracking_criteria)

    @classmethod
    def is_deteriorating(cls, ctrl_measures, patience,
                         compare_func: lambda x, y: x < y):
        """Checks if a sequence of given metrics is deteriorating
        in the last 'patience' time-steps when measured with the
        'compare_func' criteria of deterioration
        (by default an increase in the value. i.e.: increase in loss)

        Args:
            ctrl_measures (list or np.array): array of numeric measures
            patience (int): number of time-steps
            compare_func (function): comparison criteria among two consecurive
            measurements

        Returns:
            bool: whether the measure is deteriorating or not
        """
        if len(ctrl_measures) >= patience:
            return all([compare_func(a, b)
                        for a, b in zip(
                            ctrl_measures[:-1],
                            ctrl_measures[1:])])
            # logger.warning(
            #     "Early stopping due to {} deterioration"
            #     " for the last {} evaluations"
            #     "".format(metric_name, patience))
        return False

    @classmethod
    def is_stagnated(cls, ctrl_measures, patience, tolerance=1e-4):
        """Check whether a numeric measure sequence has been stagnated
        under a given 'tolerance' for a 'patience' number of steps.

        Args:
            ctrl_measures (list or np.array): array of numeric measures
            patience (int): number of time-steps
            tolerance (float, optional): absolute numeric tolerance to
            consider two measurements are close in magnitud. Defaults to 1e-4.

        Returns:
            bool: whether the measure is stagnated or not
        """
        return all([np.isclose(a, b, atol=tolerance)
                    for a, b in zip(ctrl_measures[:-1],
                                    ctrl_measures[1:])])
                # logger.warning(
                #     "Early stopping due to {} stagnation"
                #     " for the last {} evaluations"
                #     "".format(metric_name, m.patience)
                # )


class Trainer:

    # Known actions that can be taken during training
    # Expects a mapping from this actions to a trigger criteria:
    # (Criteria: MSE increases -> Action: early stop)

    def __init__(self, model, batcher, train_cfg,
                 early_stop_checker=None, lr_updater=None,
                 batch_size_updater=None, checkpointer=None):
        """Trainer helper class.
        Given a pytorch.nn.module and a data 'Batcher' train the model using
        the 'train_cfg' configuration.
        The 'model' must expose 'train_batch' and 'eval_batch' functions.
        The 'batcher' must expose 'get_training_batches' and 'get_test_batches'
        functions.

        Args:
            model (nn.Module): model to be trained
            batcher (batcher): Batcher object yielding training
                               and evaluation batches.
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

    # TODO:. Accept a function checking metric criteria to update lr,
    # stop training, increase batch size etc
    def train(self):
        """
        Trains the given model with the batches provided by the batcher.

        Returns:
            trained model
        """
        tracking_metrics = defaultdict([])
        epoch_loss = float('inf')
        train_loss = 0
        epoch_loss = 0

        self.epochs_it = tqdm(range(self.train_cfg.num_epochs))
        for epoch in self.epochs_it:
            # train the entire epoch
            epoch_loss = self._train_epoch(epoch, epoch_loss)

            # evaluate on the entire evaluation set and save model
            if epoch % self.train_cfg.eval_every == 0:
                val_metrics = self._eval(epoch, val_metrics)
                for k, v in val_metrics.items():
                    tracking_metrics[k].append(v)

                # check if early stopping
                if self.early_stop_checker and \
                        self.early_stop_checker.check(tracking_metrics):
                    break

                # check learning-rate updater
                if self.lr_updater and \
                        self.lr_updater.check(tracking_metrics):
                    logger.warning("Here would be a LR update!")

            # TensorBoard logging
            self._log(val_metrics, epoch)

            train_loss += epoch_loss

        # check if we need to save the model
        logger.warning("No checkpointing mechanism!")

        train_loss /= self.train_cfg.num_epochs
        return {
            'val_loss': val_metrics['val_loss'],
            'val_acc': val_metrics['val_acc'],
            'train_loss': train_loss
        }

    def _train_epoch(self, epoch, epoch_loss):
        self.model.train()
        self.epochs_it.set_description(
            "epoch {} - loss: {:.3f}".format(epoch, epoch_loss))

        # iterate over all batches
        for b, train_batch in enumerate(tqdm(
                self.batcher.get_training_batches(self.train_cfg.batch_size))):
            batch_x, batch_y = train_batch
            epoch_loss += self.model.train_batch(
                batch_x,
                batch_y,
                use_cuda=self.train_cfg.use_cuda
            )
        epoch_loss /= (b + 1)
        return epoch_loss

    def _eval(self, epoch):
        """Evaluate the model on the entire evaluation set."""
        self.model.eval()
        val_metrics = defaultdict(int)

        for b, val_batch in enumerate(tqdm(
                self.batcher.get_val_batch(self.train_cfg.batch_size))):
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
        return val_metrics

    def _save_model(self, epoch, control_metric, best_measure):
        """Builds the saving name and path and stores the current
        model as a checkpoint.
        """
        checkpoint_name = "{}_epoch={}_{}={:.3f}".format(
            self.train_cfg.save_name, epoch,
            control_metric, best_measure
        )
        tqdm.write('Saving model as: {}'.format(checkpoint_name))
        save_model(self.model,
                   self.train_cfg.checkpoint_path,
                   checkpoint_name)

    def _log(self, metrics, epoch):
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
