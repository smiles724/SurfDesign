from contextlib import contextmanager

from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric, SumMetric

from src.tasks._base import TASK_REGISTRY
from src.tasks.cmlm import *


@contextmanager
def on_prediction_mode(pl_module: LightningModule, enable=True):
    if not enable:
        yield
        return

    _methods = ['{}_step', '{}_step_end', '{}_epoch_end', ]

    for _method in _methods:
        _test_method, _predict_method = _method.format('test'), _method.format('predict')

        _test_method_obj = getattr(pl_module, _test_method, None)
        _predict_method_obj = getattr(pl_module, _predict_method, None)

        # swap test and predict method/hook
        setattr(pl_module, _test_method, _predict_method_obj)
        setattr(pl_module, _predict_method, _test_method_obj)

    yield

    for _method in _methods:
        _test_method, _predict_method = _method.format('test'), _method.format('predict')

        _test_method_obj = getattr(pl_module, _test_method, None)
        _predict_method_obj = getattr(pl_module, _predict_method, None)

        # swap test and predict method/hook
        setattr(pl_module, _test_method, _predict_method_obj)
        setattr(pl_module, _predict_method, _test_method_obj)


class AutoMetric(nn.Module):
    _type_shortnames = dict(mean=MeanMetric, sum=SumMetric, max=MaxMetric, min=MinMetric, )

    def __init__(self) -> None:
        super().__init__()
        self.register_parameter('_device', torch.zeros(1))

    @property
    def device(self):
        return self._device.device

    def update(self, name, value, type='mean', **kwds):
        if not hasattr(self, name):
            if isinstance(type, str):
                type = self._type_shortnames[type]
            setattr(self, name, type(**kwds))
            getattr(self, name).to(self.device)
        getattr(self, name).update(value)

    def compute(self, name):
        return getattr(self, name).compute()

    def reset(self, name):
        getattr(self, name).reset()

