"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

from .base_dataset import BaseDataset

class FullDataset(torch.utils.data.Dataset):
    """Mixed dataset with data from all available datasets."""
    
    def __init__(self, options):
        super(FullDataset, self).__init__()
        self.h36m_dataset = BaseDataset(options, 'h36m')
        self.lsp_dataset = BaseDataset(options, 'lsp-orig')
        self.coco_dataset = BaseDataset(options, 'coco')
        self.mpii_dataset = BaseDataset(options, 'mpii')
        self.up3d_dataset = BaseDataset(options, 'up-3d')
        self.length = max(len(self.h36m_dataset),
                          len(self.lsp_dataset),
                          len(self.coco_dataset),
                          len(self.mpii_dataset),
                          len(self.up3d_dataset))
        # Define probability of sampling from each detaset
        self.partition = np.array([.3, .1, .2, .2, .2]).cumsum()

    def __getitem__(self, i):
        p = np.random.rand()
        # Randomly choose element from each of the datasets according to the predefined probabilities
        if p <= self.partition[0]:
            return self.h36m_dataset[i % len(self.h36m_dataset)]
        elif p <= self.partition[1]:
            return self.lsp_dataset[i % len(self.lsp_dataset)]
        elif p <= self.partition[2]:
            return self.coco_dataset[i % len(self.coco_dataset)]
        elif p <= self.partition[3]:
            return self.mpii_dataset[i % len(self.mpii_dataset)]
        elif p <= self.partition[4]:
            return self.up3d_dataset[i % len(self.up3d_dataset)]

    def __len__(self):
        return self.length

class ITWDataset(torch.utils.data.Dataset):
    """Mixed dataset with data only from "in-the-wild" datasets (no data from H36M)."""
    
    def __init__(self, options):
        super(ITWDataset, self).__init__()
        self.lsp_dataset = BaseDataset(options, 'lsp-orig')
        self.coco_dataset = BaseDataset(options, 'coco')
        self.mpii_dataset = BaseDataset(options, 'mpii')
        self.up3d_dataset = BaseDataset(options, 'up-3d')
        self.length = max(len(self.lsp_dataset),
                          len(self.coco_dataset),
                          len(self.mpii_dataset),
                          len(self.up3d_dataset))
        # Define probability of sampling from each detaset
        self.partition = np.array([.1, .3, .3, .3]).cumsum()

    def __getitem__(self, i):
        p = np.random.rand()
        # Randomly choose element from each of the datasets according to the predefined probabilities
        if p <= self.partition[0]:
            return self.lsp_dataset[i % len(self.lsp_dataset)]
        elif p <= self.partition[1]:
            return self.coco_dataset[i % len(self.coco_dataset)]
        elif p <= self.partition[2]:
            return self.mpii_dataset[i % len(self.mpii_dataset)]
        elif p <= self.partition[3]:
            return self.up3d_dataset[i % len(self.up3d_dataset)]

    def __len__(self):
        return self.length

def create_dataset(dataset, options):
    if dataset == 'all':
        return FullDataset(options)
    elif dataset == 'itw':
        return ITWDataset(options)
    else:
        raise ValueError('Unknown dataset')
