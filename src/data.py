'''
    Define train/test loaders for experiments.
'''
import torch
from torchvision import datasets, transforms


def get_train_test_kwargs(args):
    train_kwargs = {'batch_size': args['batch_size'],
                    'shuffle': args['shuffle_batches_every_epoch']}
    test_kwargs = {'batch_size': args['test_batch_size']}
    if args['use_cuda']:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    return train_kwargs, test_kwargs


def mnist(args):
    train_kwargs, test_kwargs = get_train_test_kwargs(args)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    dataset1 = datasets.MNIST(args['data_dir'], train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST(args['data_dir'], train=False,
                       transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    args['Dataset'] = 'MNIST'
    return train_loader, test_loader