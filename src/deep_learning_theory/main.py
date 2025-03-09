import hydra
import torch
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from deep_learning_theory.callbacks import GradientNormMonitor
from deep_learning_theory.utils import flatten_dict

# Models:
# - MLP
# - Linear network
# - VGG
#
# Datasets
# - CIFAR-10 5k subset
#
# Losess
# - MSE


def store_config_in_loggers(trainer: Trainer, cfg: DictConfig) -> None:
    regular_dict = OmegaConf.to_container(cfg, resolve=True)
    flat_dict = flatten_dict(regular_dict)
    for _logger in trainer.loggers:
        if isinstance(_logger, WandbLogger):
            _logger.experiment.config.update(flat_dict)
        else:
            _logger.log_hyperparams({"config": flat_dict})


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
        ],
    )
    store_config_in_loggers(trainer, cfg)
    trainer.fit(model=module, datamodule=datamodule)


if __name__ == "__main__":
    main()
