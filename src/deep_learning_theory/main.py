from typing import Any, Iterator

import hydra
import torch
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import os
import numpy as np
from pyhessian import hessian
from torch.utils.data import DataLoader

from deep_learning_theory.callbacks import GradientNormMonitor
from deep_learning_theory.utils import flatten_dict
from tqdm.auto import tqdm

# Models:
# - Linear network
# - VGG


def store_config_in_loggers(trainer: Trainer, cfg: DictConfig) -> None:
    regular_dict = OmegaConf.to_container(cfg, resolve=True)
    flat_dict = flatten_dict(regular_dict)
    for _logger in trainer.loggers:
        if isinstance(_logger, WandbLogger):
            _logger.experiment.config.update(flat_dict)
        else:
            _logger.log_hyperparams({"config": flat_dict})


def update_model_weights_from_ckpt(
    checkpoint_path: str, module: LightningModule
) -> tuple[int, int]:
    ckpt = torch.load(checkpoint_path)
    module.load_state_dict(ckpt["state_dict"])
    return ckpt["epoch"], ckpt["global_step"]


def collect_hessian_stats(
    dir_path: str, module: LightningModule, dataloader: list[Any]
) -> Iterator[tuple[Any, ...]]:
    max_iter = 1000
    tol = 1e-5

    files = sorted(os.listdir(dir_path))
    files = list(filter(lambda x: x.endswith("ckpt"), files))
    for filename in tqdm(files, desc="Gathering Hessian stats"):
        path = os.path.join(dir_path, filename)
        ckpt = torch.load(path)
        module.load_state_dict(ckpt["state_dict"])
        global_step = ckpt["global_step"]

        hessian_comp = hessian(
            model=module.backbone,
            criterion=module.loss,
            dataloader=dataloader,
            device=module.device
        )
        top_eigenvalues, _ = hessian_comp.eigenvalues(
            top_n=1, maxIter=max_iter, tol=tol
        )
        trace = np.mean(hessian_comp.trace(maxIter=max_iter, tol=tol))
        yield {"top_eigenvalue": top_eigenvalues[0]}, {"trace": trace}, global_step


@hydra.main(config_path="conf", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    logger.info(f"\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # Reproducibility.
    torch.backends.cudnn.deterministic = True
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    module: LightningModule = hydra.utils.instantiate(cfg.module)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[
            LearningRateMonitor(
                logging_interval="step",
                log_momentum=True,
                log_weight_decay=True,
            ),
            GradientNormMonitor(),
            ModelCheckpoint(
                save_weights_only=True,
                save_on_train_epoch_end=True,
                save_top_k=-1,
                every_n_train_steps=cfg.ckpt_every_n_train_steps,
            ),
        ],
        enable_checkpointing=True,
    )
    store_config_in_loggers(trainer, cfg)
    trainer.fit(model=module, datamodule=datamodule)

    ckpt_dir_path = trainer.checkpoint_callback.dirpath

    data = []
    for batch in datamodule.train_dataloader():
        data.append(batch)

    for top_eigen, trace, global_step in collect_hessian_stats(
        ckpt_dir_path, module, data
    ):
        for _logger in trainer.loggers:
            _logger.log_metrics(top_eigen, step=global_step)
            _logger.log_metrics(trace, step=global_step)


if __name__ == "__main__":
    # Activate tensor cores on GPUS > P100.
    torch.set_float32_matmul_precision("high")
    main()
