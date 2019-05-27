import pytest
from dcscn.training.checkpointer import Checkpointer
from dcscn.training.metrics import MetricTracker


@pytest.mark.parametrize('metrics,epoch', [
    ({'mse': [1, 2, 3], 'psnr': [1, 2, 3]}, 10)
])
def test_checkpoints_metric_tracker(metrics, epoch):
    psnr_tracker = MetricTracker('psnr', MetricTracker.is_max, {})
    checkpointer = Checkpointer(
        "save_test_epoch={epoch}_psnr={psnr}",
        metric_tracker=psnr_tracker
    )

    save_string = checkpointer.format_ckpt_name(metrics=metrics, epoch=epoch)
    assert save_string == "save_test_epoch=10_psnr=3"
    assert checkpointer.check(metrics=metrics, epoch=epoch)


@pytest.mark.parametrize('metrics,epoch', [
    ({'mse': [1, 2, 3], 'psnr': [1, 2, 3]}, 2),
    ({'mse': [1, 2, 3], 'psnr': [1, 2, 3]}, 5),
    ({'mse': [1, 2, 3], 'psnr': [1, 2, 3]}, 10),
])
def test_checkpoints_interval(metrics, epoch):

    checkpointer = Checkpointer(
        "save_test_epoch={epoch}_psnr={psnr}",
        save_every=2
    )

    should_save = checkpointer.check(metrics=metrics, epoch=epoch)
    save_string = checkpointer.format_ckpt_name(metrics=metrics,
                                                epoch=epoch)
    assert save_string == "save_test_epoch={}_psnr=3".format(epoch)
    assert (not should_save) == epoch % 2
