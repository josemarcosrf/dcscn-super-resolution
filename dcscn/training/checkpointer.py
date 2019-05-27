

class Checkpointer():
    """Simple checkpointer class.
    Offers two functions:
    'check' to check if a model checkpoint should be created based
    either on epochs intervals or by tracking an specific metric.
    'formt_ckpt_name' formats a given string to use as checkpoint name
    allowing it to be formated with the metrics provided by the model
    evaluation function, epoch or other training paramters.
    """

    def __init__(self, save_path, ckpt_name_template,
                 save_every=None, metric_tracker=None):
        self.checkpoint_path = save_path
        self.ckpt_name_template = ckpt_name_template
        self.save_every = save_every
        self.metric_tracker = metric_tracker

    def check(self, metrics, epoch):
        if self.save_every:
            return epoch % self.save_every == 0

        return self.metric_tracker.check(metrics)

    def format_ckpt_name(self, epoch, metrics):
        """ Makes available to the format string all the latest available
        measurements, epoch and training status.
        """
        last_measures = dict([(k, v[-1]) for k, v in metrics.items()])
        return self.ckpt_name_template.format(**last_measures, epoch=epoch)

