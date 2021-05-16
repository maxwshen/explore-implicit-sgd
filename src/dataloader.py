'''
'''

import torch
from torch.utils.data import Dataset, DataLoader


class SubsetBatchDataset(Dataset):
    '''
        For computing Hessian on specific subset of minibatches.
    '''
    def __init__(self, batches: list):
        self.data = []
        self.target = []
        for batch in batches:
            data, target = batch
            self.data.append(data)
            self.target.append(target)
        self.data = torch.cat(self.data)
        self.target = torch.cat(self.target)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def get_subset_batch_loader(batches, args):
    return DataLoader(SubsetBatchDataset(batches),
            batch_size=args['batch_size'], shuffle=False)