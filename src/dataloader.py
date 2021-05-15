'''
'''

import torch
from torch.utils.data import Dataset, DataLoader

class SingleBatchLoader(Dataset):
    '''
        To compute Hessian on a single minibatch, we need to construct a small
        Dataloader that only returns that batch.
    '''
    def __init__(self, batch):
        data, target = batch
        self.data = data
        self.target = target


    def __len__(self):
        return len(self.target)


    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def get_single_batch_loader(batch):
    sbl = SingleBatchLoader(batch)
    return DataLoader(sbl)