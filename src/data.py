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


def cifar10(args):
    train_kwargs, test_kwargs = get_train_test_kwargs(args)

    # Add data augmentation for training data
    normalize_mean = (0.4914, 0.4822, 0.4465)
    normalize_std = (0.2471, 0.2435, 0.2616)
    transform_train = transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(normalize_mean, normalize_std)])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)])

    dataset1 = datasets.CIFAR10(root=args['data_dir'], train=True,
                                            download=True, transform=transform_train)
    dataset2 = datasets.CIFAR10(root=args['data_dir'], train=False,
                                          download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    args['Dataset'] = 'CIFAR-10'
    return train_loader, test_loader