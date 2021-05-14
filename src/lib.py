'''
'''
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pandas as pd
import os
import datetime


'''
    Gradients 
'''
def gradient_norm(model):
    '''
        Requires loss.backward() first
    '''
    norm = [p.grad.data.norm(2).item()**2 for p in model.parameters()]
    return sum(norm)**(1.0/2)


def gradient(model):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    # print(grads.shape)
    return torch.Tensor(grads)


'''
    Stats
'''
def calc_grad_var(model, device, train_loader, samplesize = 5):
    '''
      Calculate variance of gradients for NLL loss function.
      Uses low-rank approximation to Hessian.
      Can be used as loss: Backprop-able
    '''
    dd = defaultdict(list)
    for data, target in iter_sample_fast(train_loader, samplesize):
        data, target = data.to(device), target.to(device)
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()

        dd['Grad'].append(gradient(model))
    
    grads = torch.Tensor(dd['Grad'])

    mean_grad = torch.mean(grads, axis=0)
    assert mean_grad.shape == dd['Grad'][0].shape
    grad_var = sum([torch.linalg.norm(g - mean_grad, ord=2)**2 for g in dd['Grad']])
    return grad_var


def calc_loss_var(model, device, train_loader, samplesize = 5):
    '''
      Calculate variance of loss for NLL loss function.
      Can be used as loss: Backprop-able
    '''
    dd = defaultdict(list)
    for data, target in iter_sample_fast(train_loader, samplesize):
        data, target = data.to(device), target.to(device)
        output = model(data)

        loss = F.nll_loss(output, target)
        dd['Loss'].append(loss)
    
    losses = torch.stack(dd['Loss'])
    return torch.var(losses, 0)


def get_stats(model, device, train_loader, samplesize = 5):
    '''
      Compute, across minibatches:
      - Loss
      - Gradient norm
      Return statistics on both, in particular the variance.

      A bit slow to do across all batches; subset batches instead?
    '''
    model.train()
    dd = defaultdict(list)
    for data, target in iter_sample_fast(train_loader, samplesize):
        data, target = data.to(device), target.to(device)
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()

        dd['loss'].append(loss.item())
        dd['grad norm'].append(gradient_norm(model))
        grad = np.array(gradient(model).cpu())
        dd['grad'].append(grad)
    
    losses = np.array(dd['loss'])
    grad_norms = np.array(dd['grad norm'])

    mean_grad = np.mean(dd['grad'], axis=0)
    assert mean_grad.shape == dd['grad'][0].shape
    grad_var = sum([np.linalg.norm(g - mean_grad, 2)**2 for g in dd['Grad']])

    stats_d = {
        'loss_mean': np.mean(losses), 
        'loss_var': np.var(losses), 
        'gradnorm_mean': np.mean(grad_norms),
        'gradnorm_var': np.var(grad_norms),
        'grad_var': grad_var,
    }
    for k, v in stats_d:
        print(f'{k}:\t{v}')
    return stats_d


'''
    Support
'''
def iter_sample_fast(iterable, samplesize):
    '''
        Randomly sample from an iterator efficiently
    '''
    results = []
    iterator = iter(iterable)
    # Fill in the first samplesize elements:
    try:
        for _ in range(samplesize):
            results.append(iterator.next())
    except StopIteration:
        raise ValueError("Sample larger than population.")
    random.shuffle(results)  # Randomize their positions
    for i, v in enumerate(iterator, samplesize):
        r = random.randint(0, i)
        if r < samplesize:
            results[r] = v  # at a decreasing rate, replace random items
    return results


def write_args(args):
    with open(args['results_dir'] + '/args.txt', 'w'):
        f.write(str(datetime.datetime.now()) + '\n')
        for k, v in args:
            f.write(f'{k}: {v}\n')
    return