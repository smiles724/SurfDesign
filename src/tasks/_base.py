from typing import Any, List, Union
from typing import Callable, Dict, Optional

import torch
from pytorch_lightning import LightningModule
from torch import distributed as dist
from torch import nn

from src import utils
from src.utils.lr_scheduler import get_scheduler
from src.utils.optim import get_optimizer

TASK_REGISTRY = {}
log = utils.get_logger(__name__)


def register_task(name):
    def decorator(cls):
        cls._name_ = name
        TASK_REGISTRY[name] = cls
        return cls

    return decorator


class TaskLitModule(LightningModule):
    """Example of LightningModule for seq2seq learning.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self, model: List[nn.Module], criterion: nn.Module = None, optimizer: Union[Callable, torch.optim.Optimizer] = None,
                 lr_scheduler: Union[Callable, torch.optim.lr_scheduler._LRScheduler] = None, ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.criterion = criterion   # loss function
        self.valid_logged = {}

    def setup(self, stage=None) -> None:
        self._stage = stage
        super().setup(stage)

    @property
    def lrate(self):
        for param_group in self.trainer.optimizers[0].param_groups:
            return param_group['lr']

    @property
    def stage(self):
        return self._stage

    def log(self, name: str, value, prog_bar: bool = False, logger: bool = True, on_step: Optional[bool] = None, on_epoch: Optional[bool] = None,
            **kwargs) -> None:
        if on_epoch and not self.training:
            self.valid_logged[name] = value
        return super().log(name, value, prog_bar, logger, on_step, on_epoch, **kwargs)

    # -------# Training #-------- #
    def step(self, batch):
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def on_train_step_end(self, step_output: Union[torch.Tensor, Dict[str, Any]]) -> Union[torch.Tensor, Dict[str, Any]]:
        return super().training_step_end(step_output)

    def on_train_epoch_end(self, outputs: List[Any]):
        pass

    # -------# Evaluating #-------- #
    def validation_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def on_validation_step_end(self, *args, **kwargs) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        return super().on_validation_step_end(*args, **kwargs)

    def on_validation_epoch_end(self, outputs: List[Any]):
        logging_info = ", ".join(f"{key}={val:.3f}" for key, val in self.valid_logged.items())
        logging_info = f"Validation Info @ (Epoch {self.current_epoch}, global step {self.global_step}): {logging_info}"
        log.info(logging_info)

    def test_step(self, batch: Any, batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def on_test_step_end(self, *args, **kwargs) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        return self.validation_step_end(*args, **kwargs)

    def on_test_epoch_end(self, outputs: List[Any]):
        return self.on_validation_epoch_end(outputs)

    # -------# Inference/Prediction #-------- #
    def forward(self, batch):
        raise NotImplementedError

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        raise NotImplementedError

    def predict_epoch_end(self, results: List[Any]) -> None:
        raise NotImplementedError

    # -------# Optimizers & Lr Schedulers #-------- #
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = get_optimizer(self.hparams.optimizer, self.parameters())
        if 'lr_scheduler' in self.hparams and self.hparams.lr_scheduler is not None:
            lr_scheduler, extra_kwargs = get_scheduler(self.hparams.lr_scheduler, optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': {"scheduler": lr_scheduler, **extra_kwargs}}
        return optimizer

    # -------# Others #-------- #
    def on_train_epoch_end(self) -> None:
        if dist.is_initialized() and hasattr(self.trainer.datamodule, 'train_batch_sampler'):
            self.trainer.datamodule.train_batch_sampler.set_epoch(self.current_epoch + 1)
            self.trainer.datamodule.train_batch_sampler._build_batches()