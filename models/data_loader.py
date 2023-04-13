import random
import os
from torchvision import datasets
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class MNISTDataset(Dataset):

    def __init__(self, data_dir, test):

        if test != True:
            train_val_dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=transforms.ToTensor())

            # normalize them with their mean and std so that our loss function converges fast
            imgs = torch.stack([img for img, _ in train_val_dataset], dim=0)

            mean = imgs.view(1, -1).mean(dim=1)    # or imgs.mean()
            std = imgs.view(1, -1).std(dim=1)     # or imgs.std()
            mean, std

            mnist_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            train_val_dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=mnist_transforms)
    
            train_size = int(0.9 * len(train_val_dataset))
            val_size = len(train_val_dataset) - train_size

            self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset=train_val_dataset, lengths=[train_size, val_size])

        
        else:
            self.test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=mnist_transforms)

class CIFAR10Dataset(Dataset):

    def __init__(self, data_dir, test):

        if test != True:
            train_val_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transforms.ToTensor())

            # normalize them with their mean and std so that our loss function converges fast
            imgs = torch.stack([img for img, _ in train_val_dataset], dim=0)

            mean = imgs.view(1, -1).mean(dim=1)    # or imgs.mean()
            std = imgs.view(1, -1).std(dim=1)     # or imgs.std()
            mean, std

            cifar10_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            train_val_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=cifar10_transforms)


            train_size = int(0.9 * len(train_val_dataset))
            val_size = len(train_val_dataset) - train_size

            self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset=train_val_dataset, lengths=[train_size, val_size])

        
        else:
            self.test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transforms.ToTensor())


def fetch_dataloader(types, data_dir, dataset, params):

    dataloaders = {}

    if dataset == 'MNIST':

        if 'test' in types:
            test_dataset = MNISTDataset(data_dir, test=True)
            dataloaders['test'] = DataLoader(dataset=test_dataset, batch_size=params.batch_size, shuffle=True)
        else:
            full_data_set = MNISTDataset(data_dir, test=False)
            train_dataset, val_dataset = full_data_set.train_dataset, full_data_set.val_dataset
            dataloaders['train'] = DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=True)
            dataloaders['val'] = DataLoader(dataset=val_dataset, batch_size=params.batch_size, shuffle=True) 
    else:
        print("Using CIFAR10 dataset\n")
        if 'test' in types:
            test_dataset = CIFAR10Dataset(data_dir, test=True)
            dataloaders['test'] = DataLoader(dataset=test_dataset, batch_size=params.batch_size, shuffle=True)
        else:
            full_data_set = CIFAR10Dataset(data_dir, test=False)
            train_dataset, val_dataset = full_data_set.train_dataset, full_data_set.val_dataset
            dataloaders['train'] = DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=False)
            dataloaders['val'] = DataLoader(dataset=val_dataset, batch_size=params.batch_size, shuffle=False) 

    return dataloaders