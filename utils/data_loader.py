import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from utils.fashion_mnist import MNIST, FashionMNIST
import scipy.io
import numpy as np

def get_data_loader(args):

    if args.dataset == 'mnist':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        train_dataset = MNIST(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = MNIST(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'alloys_c':

        train_dataset= scipy.io.loadmat(args.dataroot+'ZrBMG_CD1.mat')
        train_dataset= np.transpose(train_dataset['V4D'])
        train_dataset_2 = scipy.io.loadmat(args.dataroot+'ZrBMG_CD2.mat')
        train_dataset_2 = np.transpose(train_dataset_2['V4D'])[:,:,:,8:]
        train_dataset = np.concatenate((train_dataset,train_dataset_2),axis=3)
        train_dataset = np.abs(train_dataset)
        # x_train_dataset = train_dataset[:,:,:,:8]
        # y_train_dataset = train_dataset[:,:,:,8:]

        test_dataset = train_dataset[:100,:,:,:]

        # test_dataset = scipy.io.loadmat(args.dataroot)
        # test_dataset = np.transpose(test_dataset['V4D'])
        
        # x_train_dataloader = data_utils.DataLoader(x_train_dataset, batch_size=args.batch_size, shuffle=True)
        # y_train_dataloader = data_utils.DataLoader(y_train_dataset, batch_size=args.batch_size, shuffle=True)
        # test_dataloader = data_utils.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True)

        # return x_train_dataloader, y_train_dataloader,test_dataloader
    
    elif args.dataset == 'alloys':

        train_dataset= scipy.io.loadmat(args.dataroot+'hybridBMG_D1.mat')
        train_dataset= np.transpose(train_dataset['V4D'])
        # train_dataset_2 = scipy.io.loadmat(args.dataroot+'ZrBMG_CD2.mat')
        # train_dataset_2 = np.transpose(train_dataset_2['V4D'])[:,:,:,8:]
        # train_dataset = np.concatenate((train_dataset,train_dataset_2),axis=3)
        # train_dataset = np.abs(train_dataset)
        

        # x_train_dataset = train_dataset[:,:,:,:8]
        # y_train_dataset = train_dataset[:,:,:,8:]
        test_dataset = train_dataset

        # test_dataset = scipy.io.loadmat(args.dataroot)
        # test_dataset = np.transpose(test_dataset['V4D'])
        
        # x_train_dataloader = data_utils.DataLoader(x_train_dataset, batch_size=args.batch_size, shuffle=True)
        # y_train_dataloader = data_utils.DataLoader(y_train_dataset, batch_size=args.batch_size, shuffle=True)
        # test_dataloader = data_utils.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True)

        # return x_train_dataloader, y_train_dataloader,test_dataloader


    elif args.dataset == 'fashion-mnist':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        train_dataset = FashionMNIST(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = FashionMNIST(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'cifar':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = dset.CIFAR10(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = dset.CIFAR10(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'stl10':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        train_dataset = dset.STL10(root=args.dataroot, split='train', download=args.download, transform=trans)
        test_dataset = dset.STL10(root=args.dataroot,  split='test', download=args.download, transform=trans)

    # Check if everything is ok with loading datasets
    # assert train_dataset
    # assert test_dataset

    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = data_utils.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader
