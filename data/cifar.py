"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import os
import pickle
import gzip
from tkinter import NONE
from urllib.request import urlretrieve
import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.mypath import MyPath

class CIFAR10(Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``mnist`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://yann.lecun.com/exdb/mnist/'
    train_list = ['train-images-idx3-ubyte.gz',
             'train-labels-idx1-ubyte.gz']
    test_list = ['t10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']

    def __init__(self, root=MyPath.db_root_dir('cifar-10'), train=True, transform=None, 
                    download=False):

        super(CIFAR10, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        
        # dowload missing files
        for file in self.train_list + self.test_list:
            if file not in os.listdir(root):
                urlretrieve(self.url + file, os.path.join(root, file))
                print("Downloaded %s to %s" % (file, root))

        # train / test split
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        
        # now load the picked numpy arrays
        self.data = None
        self.targets = None

        with gzip.open(os.path.join(root, downloaded_list[0])) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            self.data = np.frombuffer(f.read(), 'B', offset=16)
        self.data = self.data.reshape(-1, 28, 28,1)

        with gzip.open(os.path.join(root, downloaded_list[1])) as f:
            # First 8 bytes are magic_number, n_labels
            self.targets = np.frombuffer(f.read(), 'B', offset=8)

        # names of classes
        self.classes = [str(i) for i in range(10)]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        H = 32; W = 32

        img, target = self.data[index], self.targets[index]
        img_size = (H, W)
        img = np.concatenate([img]*3, axis=2)
        img = Image.fromarray(img)      
        img = img.resize(img_size)

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': int(target), 'meta': {'im_size': img_size, 'index': index, 'class_name': str(target)}}
        
        return out

    def get_image(self, index):
        H = 32; W = 32

        img = self.data[index]
        img_size = (H, W)
        img = np.concatenate([img]*3, axis=2)
        img = Image.fromarray(img)      
        img = img.resize(img_size)

        return np.array(img)
        
    def __len__(self):
        return self.data.shape[0]

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")