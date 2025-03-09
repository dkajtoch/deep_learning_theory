from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from lightning import Callback
from lightning.fabric.utilities.exceptions import MisconfigurationException
from torch.optim import Optimizer
from typing_extensions import override

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer


class GradientNormMonitor(Callback):

    @override
    def on_train_start(self, trainer: "Trainer", *args: Any, **kwargs: Any) -> None:
        if not trainer.loggers:
            raise MisconfigurationException(
                "Cannot use `GradientNormMonitor` callback with `Trainer` "
                "that has no logger."
            )

    @override
    def on_before_optimizer_step(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        optimizer: Optimizer,
    ) -> None:
        if not trainer._logger_connector.should_update_logs:
            return

        grad_norm = self.gather_grad_norms(trainer.lightning_module)
        for logger in trainer.loggers:
            logger.log_metrics(
                {"train/grad_norm": grad_norm},
                step=trainer.fit_loop.epoch_loop._batches_that_stepped,
            )

    @staticmethod
    def gather_grad_norms(module: nn.Module) -> float:
        total_norm = torch.tensor(0.0, device=module.device)
        for param in module.parameters():
            if param.grad is not None:
                param_norm = param.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        res: float = (total_norm ** (1.0 / 2)).cpu().item()
        return res
