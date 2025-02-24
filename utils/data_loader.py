import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import os
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
import pandas as pd
import numpy as np

class AlloyDataset(Dataset):
    """Alloy Dataset.

    Args:
        root (string): Root directory of dataset where the CSV file exists.
        train (bool, optional): If True, creates dataset from training set,
            otherwise from test set.
        split (int, optional): Number of samples in the test set.
        transform (callable, optional): A function/transform that takes in a sample
            and returns a transformed version.
    """
    def __init__(self, args, root, train=True, split=10, transform=None):
        self.root = root
        self.train = train
        self.split = split
        self.transform = transform

        # Load the dataset
        dataset = pd.read_csv(args.dataroot+'Alloy_train.csv')
        dataset.drop("source", axis=1, inplace=True)
        dataset = dataset.values

        # Split the dataset into training and testing sets
        if self.train:
            self.data = dataset[:, :]
        else:
            self.data = dataset[:, :]

    def __getitem__(self, index):
        sample = self.data[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data)
    
def get_data_loader(args):
    if args.dataset == 'alloys':

        train_dataset = AlloyDataset(args, root=args.dataroot, train=True)
        test_dataset = AlloyDataset(args, root=args.dataroot, train=False)


    # Check if everything is ok with loading datasets
    # assert train_dataset
    # assert test_dataset

    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = data_utils.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader
