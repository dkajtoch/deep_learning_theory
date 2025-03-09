import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torchmetrics.classification import Accuracy


class VisionClassificationModule(LightningModule):
    train_acc: Accuracy
    valid_acc: Accuracy

    def __init__(
        self,
        backbone: nn.Module,
        optimizer: functools.partial,
        lr_scheduler: functools.partial,
        lr_scheduler_interval: str = "step",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self._partial_optimizer = optimizer
        self._partial_lr_scheduler = lr_scheduler
        self._lr_scheduler_interval = lr_scheduler_interval

    def configure_model(self) -> None:
        # Torch compile influences reproducibility. Use either
        # compiled or non-compiled model.
        self.backbone = torch.compile(self.backbone)  # type: ignore[assignment]

    def _attach_metrics(self) -> None:
        assert hasattr(self.trainer, "datamodule")
        num_classes = self.trainer.datamodule.num_classes
        self.train_acc = Accuracy("multiclass", num_classes=num_classes)
        self.valid_acc = Accuracy("multiclass", num_classes=num_classes)

    def setup(self, stage: str) -> None:
        self._attach_metrics()

    def forward(self, images: Tensor) -> Tensor:
        preds: Tensor = self.backbone(images)
        return preds

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        self.log("train/loss", loss)

        self.train_acc(logits, labels)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        self.log("val/loss", loss, on_epoch=True, on_step=True)

        self.valid_acc(logits, labels)
        self.log("val/acc", self.valid_acc, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        bias_group = []
        weight_group = []
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                if "bias" in name:
                    bias_group.append(param)
                else:
                    weight_group.append(param)

        optim_params = self._partial_optimizer.keywords.copy()
        lr = optim_params.pop("lr")
        weight_decay = optim_params.pop("weight_decay", 0)

        optimizer = self._partial_optimizer(
            [
                {
                    "params": weight_group,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    **optim_params,
                },
                {"params": bias_group, "lr": lr * 2, "weight_decay": 0, **optim_params},
            ]
        )
        lr_scheduler = self._partial_lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self._lr_scheduler_interval,
            },
        }
