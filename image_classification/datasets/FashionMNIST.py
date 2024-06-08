import os
from typing import Tuple, Optional
import logging
import torch
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset

from transforms.image import RandomPatch
import lightning.pytorch as L


def get_default_transforms():
    """

    :return: The default train, valid and test transforms
    """
    normalize = transforms.Normalize((0.1307,), (0.3081,))  # MNIST

    train_transforms = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        RandomPatch()
    ])

    valid_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return train_transforms, valid_transforms, test_transforms


def get_datasets(root='.', train_transforms=None, valid_transforms=None, test_transforms=None, ) -> Tuple[
    Dataset, Dataset, Dataset]:
    """
    Get the train, valid and test datasets
    :param root: The dataset root, if the data don't exist, they will be downloaded there
    :param train_transforms: Specify the transformations for the train dataset. If None, defaults are used
    :param valid_transforms: Specify the transformations for the valid dataset. If None, defaults are used
    :param test_transforms: Specify the transformations for the test dataset. If None, defaults are used
    :return: the train, valid and test torch Datasets
    """
    default_transforms = get_default_transforms()
    train_transforms = train_transforms or default_transforms[0]
    valid_transforms = valid_transforms or default_transforms[1]
    test_transforms = test_transforms or default_transforms[2]

    train_dataset = FashionMNIST(
        root,
        download=True,
        train=True,
        transform=train_transforms)
    valid_dataset = FashionMNIST(
        root,
        download=True,
        train=True,
        transform=valid_transforms)
    test_dataset = FashionMNIST(
        root,
        download=True,
        train=False,
        transform=test_transforms)

    return train_dataset, valid_dataset, test_dataset


def get_dataloaders(train_dataset=None, valid_dataset=None, test_dataset=None, split_perc=0.8, **kwargs):
    """
    Get the train, valid, test dataloaders. Train/valid split is specified as percentage
    :param train_dataset: torch dataset used for training
    :param valid_dataset: torch dataset used for validation
    :param test_dataset: torch dataset used for testing
    :param split_perc: percentage of train samples in range (0, 1)
    :return: the train, valid and test torch Dataloaders
    """
    if not 0 <= split_perc <= 1:
        raise ValueError(f'split_perc should be between 0 and 1, but was {split_perc}')

    datasets = get_datasets()
    train_dataset = train_dataset or datasets[0]
    valid_dataset = valid_dataset or datasets[1]
    test_dataset = test_dataset or datasets[2]

    indices = list(range(len(train_dataset)))
    split = int(split_perc * len(train_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               sampler=SubsetRandomSampler(indices[:split]),
                                               **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               sampler=SubsetRandomSampler(indices[split:]),
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              shuffle=True,
                                              **kwargs)

    return train_loader, valid_loader, test_loader


class FashionMNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", train_split: float = 0.8, seed: int = 42, batch_sz: int = 64,
                 num_workers: Optional[int] = None):
        super().__init__()
        self.data_dir = data_dir
        self.train_transform, self.valid_transform, self.test_transform = get_default_transforms()
        self.__train_split = train_split
        self.__seed = seed
        self.__batch_sz = batch_sz
        self.__num_workers = num_workers or os.cpu_count() or 0
        logging.info(f"Using {self.__num_workers} workers.")

    def prepare_data(self):
        # download
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = FashionMNIST(self.data_dir, train=True)
            from torch.utils.data import random_split
            train_ix, val_ix = random_split(
                range(len(mnist_full)), [self.__train_split, 1 - self.__train_split],
                generator=torch.Generator().manual_seed(self.__seed)
            )
            self.mnist_train = torch.utils.data.Subset(FashionMNIST(
                self.data_dir,
                download=True,
                train=True,
                transform=self.train_transform), train_ix)
            self.mnist_val = torch.utils.data.Subset(FashionMNIST(
                self.data_dir,
                download=True,
                train=True,
                transform=self.valid_transform), val_ix)

            # Assign test dataset for use in dataloader(s)
            if stage == "test":
                self.mnist_test = FashionMNIST(self.data_dir, train=False, transform=self.test_transform)

            if stage == "predict":
                self.mnist_predict = FashionMNIST(self.data_dir, train=False, transform=self.test_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_train,
                                           batch_size=self.__batch_sz,
                                           shuffle=True,
                                           num_workers=self.__num_workers,
                                           pin_memory=True,
                                           persistent_workers=self.__num_workers > 0,
                                           drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_val,
                                           batch_size=self.__batch_sz,
                                           num_workers=self.__num_workers,
                                           pin_memory=True,
                                           persistent_workers=self.__num_workers > 0)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_test, batch_size=self.__batch_sz,
                                           num_workers=self.__num_workers,
                                           pin_memory=True,
                                           persistent_workers=self.__num_workers > 0)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_predict,
                                           batch_size=self.__batch_sz,
                                           num_workers=self.__num_workers,
                                           pin_memory=True,
                                           persistent_workers=self.__num_workers > 0)

    @property
    def n_classes(self):
        return 10

    @property
    def dataset_name(self):
        return 'FashionMNIST'
