import os
from collections.abc import Callable
from typing import Any

import numpy as np
import torchvision.transforms as transform_lib
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageNet


class VisionDataModule(LightningDataModule):
    dataset_cls: type

    dataset_train: Dataset
    dataset_val: Dataset

    def __init__(
        self,
        data_dir: str | None = "data",
        num_workers: int = 0,
        normalize: bool = True,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        train_transform: Callable | None = None,
        val_transform: Callable | None = None,
        num_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.num_samples = num_samples

    @property
    def num_classes(self):
        raise NotImplementedError

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """Saves files to data_dir."""
        self.dataset_cls(self.data_dir, train=True, download=True)
        self.dataset_cls(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        """Creates train, val, and test dataset."""
        if stage == "fit" or stage is None:
            dataset_train = self.dataset_cls(
                self.data_dir,
                train=True,
                transform=self.train_transform,
            )
            if self.num_samples is not None:
                indices = self._create_balanced_sample(
                    dataset_train, self.num_samples, self.num_classes
                )
                self.dataset_train = Subset(dataset_train, indices)

            self.dataset_val = self.dataset_cls(
                self.data_dir,
                train=False,
                transform=self.val_transform,
            )

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(
        self, *args: Any, **kwargs: Any
    ) -> DataLoader | list[DataLoader]:
        """The val dataloader."""
        return self._data_loader(self.dataset_val)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    @staticmethod
    def _create_balanced_sample(
        dataset: IterableDataset, num_samples: int, num_classes: int
    ) -> list[int]:
        rng = np.random.RandomState(42)

        samples_per_class = num_samples // num_classes
        remainder = num_samples % num_classes

        target_counts = [samples_per_class] * num_classes

        for i in range(remainder):
            target_counts[i] += 1

        class_indices: list[list[int]] = [[] for _ in range(num_classes)]

        for idx, (_, label) in enumerate(dataset):
            if isinstance(label, Tensor):
                label = label.item()
            class_indices[label].append(idx)

        selected_indices: list[int] = []
        for class_idx, target_count in enumerate(target_counts):
            available_count = len(class_indices[class_idx])
            count_to_sample = min(target_count, available_count)

            sampled_indices = rng.choice(
                class_indices[class_idx], size=count_to_sample, replace=False
            ).tolist()

            selected_indices.extend(sampled_indices)  # type: ignore[assignment,arg-type]
        rng.shuffle(selected_indices)
        return selected_indices


class CIFAR10DataModule(VisionDataModule):
    dataset_cls = CIFAR10

    def __init__(self, **kwargs):
        train_transform = transform_lib.Compose(
            [
                transform_lib.RandomCrop(32, padding=4),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.ToTensor(),
                self.caffe_equiv2_normalization(),
            ]
        )
        val_transform = transform_lib.Compose(
            [
                transform_lib.ToTensor(),
                self.caffe_equiv2_normalization(),
            ]
        )
        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            **kwargs,
        )

    @property
    def num_classes(self):
        return 10

    @staticmethod
    def default_normalization() -> Callable:
        transform: Callable = transform_lib.Normalize(
            [0.4914, 0.48216, 0.44653], [0.24703, 0.24349, 0.26159]
        )
        return transform

    @staticmethod
    def caffe_equiv1_normalization() -> Callable:
        # See: https://github.com/ethanhe42/resnet-cifar10-caffe/blob/master/resnet-56/trainval.prototxt#L10
        # Produces pixel values in range [-128, 128]
        transform: Callable = transform_lib.Normalize(
            [128.0 / 255, 128.0 / 255, 128.0 / 255], [1.0 / 255, 1.0 / 255, 1.0 / 255]
        )
        return transform

    @staticmethod
    def caffe_equiv2_normalization() -> Callable:
        # See: https://github.com/ethanhe42/resnet-cifar10-caffe/blob/master/resnet-44/trainval.prototxt#L10
        # Produces pixel values in range [-1, 1]
        transform: Callable = transform_lib.Normalize(
            [128.0 / 255, 128.0 / 255, 128.0 / 255],
            [128.0 / 255, 128.0 / 255, 128.0 / 255],
        )
        return transform


class CIFAR100DataModule(VisionDataModule):
    dataset_cls = CIFAR100

    def __init__(self, **kwargs):
        train_transform = transform_lib.Compose(
            [
                transform_lib.ToTensor(),
                self.default_normalization(),
            ]
        )
        val_transform = transform_lib.Compose(
            [
                transform_lib.ToTensor(),
                self.default_normalization(),
            ]
        )
        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            **kwargs,
        )

    @property
    def num_classes(self):
        return 100

    @staticmethod
    def default_normalization() -> Callable:
        transform: Callable = transform_lib.Normalize(
            [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        )
        return transform


class MNISTDataModule(VisionDataModule):
    dataset_cls = MNIST

    def __init__(self, **kwargs):
        train_transform = transform_lib.Compose(
            [transform_lib.ToTensor(), self.default_normalization()]
        )
        val_transform = transform_lib.Compose(
            [transform_lib.ToTensor(), self.default_normalization()]
        )
        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            **kwargs,
        )

    @property
    def num_classes(self):
        return 10

    @staticmethod
    def default_normalization() -> Callable:
        transform: Callable = transform_lib.Normalize(mean=(0.5,), std=(0.5,))
        return transform

    @staticmethod
    def standard_normalization() -> Callable:
        transform: Callable = transform_lib.Normalize(mean=(0.1307,), std=(0.3081,))
        return transform


class ImageNetDataModule(VisionDataModule):
    dataset_cls = ImageNet

    def __init__(self, **kwargs):

        train_transform = transform_lib.Compose(
            [
                transform_lib.RandomResizedCrop(224),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.ToTensor(),
                self.default_normalization(),
            ]
        )
        val_transform = transform_lib.Compose(
            [
                transform_lib.RandomResizedCrop(224),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.ToTensor(),
                self.default_normalization(),
            ]
        )

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            **kwargs,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.dataset_train = self.dataset_cls(
                self.data_dir,
                split="train",
                transform=self.train_transform,
            )
            self.dataset_val = self.dataset_cls(
                self.data_dir,
                split="val",
                transform=self.val_transform,
            )

    @staticmethod
    def default_normalization() -> Callable:
        transform: Callable = transform_lib.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        return transform
