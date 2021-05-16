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

from hessian_eigenthings import compute_hessian_eigenthings
import dataloader

'''
    Gradients and Hessians
'''
def gradient_norm(model):
    # Requires loss.backward() first
    norm = [p.grad.data.norm(2).item()**2 for p in model.parameters()]
    return torch.Tensor([sum(norm)**(1.0/2)])


def gradient(model):
    # Requires loss.backward() first
    grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads).cpu()
    return torch.Tensor(grads)


def gradient_variance(grads):
    # Input: List of gradients
    assert len(grads) > 1, 'ERROR: Attempted to take variance of <2 items'
    mean_grad = torch.mean(grads, axis=0)
    assert mean_grad.shape == grads[0].shape
    grad_var = sum([torch.linalg.norm(g - mean_grad, 2)**2 for g in grads])
    return grad_var, mean_grad


def eigen_approx_hessian(model, batches, args):
    '''
        Eigen-approximation to Hessian.
        https://github.com/noahgolmant/pytorch-hessian-eigenthings

        To compute Hessian on a single minibatch, we construct a 
        dataloader that provides only that batch.
    '''
    loader = dataloader.get_subset_batch_loader(batches, args)
    eigenvals, eigenvecs = compute_hessian_eigenthings(model,
            loader, args['loss_func'],
            num_eigenthings=args['num_eigens_hessian_approx'],
            use_gpu=args['use_cuda'],
            full_dataset=True)
    eigenvals = np.expand_dims(eigenvals, -1)
    return torch.Tensor(eigenvals.copy()), torch.Tensor(eigenvecs.copy())
    

def assign_gradient_to_model(model, gradient, device):
    '''
        Manually override gradient
    '''
    gradient = torch.Tensor(gradient)
    for param in model.parameters():
        size = np.prod(param.grad.shape)
        g = gradient[:size].to(device)
        param.grad = g.reshape(param.grad.shape)
        gradient = gradient[size:]
    assert len(gradient) == 0, 'ERROR: Gradient shape did not match num. parameters'
    return


'''
    Special gradients
'''
def grad_of_grad_var(mean_grad, dd):
    '''
        Uses fast eigenapproximation for batch-wise Hessian and mean Hessian.
        Uses proper order of matrix multiplication to avoid large memory cost.

        For B total NxN matrix with m-rank eigenapproximation,
        Time:  O(B N M^2)
        Space: O(NM)
    '''
    learning_gradient = torch.zeros(mean_grad.shape)
    mean_evals = dd['mean_eigenvalues']
    mean_evecs = dd['mean_eigenvectors']
    mean_hessian_left = mean_evecs.T @ mean_evals
    for g, evals, evecs in zip(dd['grad'], dd['eigenvalues'], dd['eigenvectors']):
        grad_diff = (g - mean_grad)
        local_term = (evecs.T @ evals) @ (evecs @ grad_diff)
        global_term = mean_hessian_left @ (mean_evecs @ grad_diff)
        lg = 2 * (local_term - global_term)
        learning_gradient += lg
    return learning_gradient


def grad_of_grad_var_memory(mean_grad, dd):
    '''
        DEPRECATED
        Explicitly stores full Hessians in memory - doesn't scale.

        For B total NxN matrix with m-rank eigenapproximation,
        Time:  O(B N^2 M + M N^2)
        Space: O(BN^2)
    '''
    hessians = [evec.T @ evals @ evec
                for evec, evals in zip(dd['eigenvectors'], dd['eigenvalues'])]
    mean_hessian = torch.mean(hessians, axis=0)
    assert mean_hessian.shape == hessians[0].shape

    learning_gradient = torch.zeros(mean_grad.shape)
    for g, h in zip(dd['grad'], hessians):
        lg = 2 * (h - mean_hessian) @ (g - mean_grad)
        learning_gradient += lg
    return learning_gradient


def compute_row_from_eigendecomp(eigenvecs, eigenvals, i):
    '''
        For NxN matrix with m-rank eigenapproximation,
        computes a (1,N) row in O(M^2 + NM^2) = O(NM^2) time.
        (1,M) (M,M) (M,N) -> (1,N)
    '''
    return eigenvecs.T[i] @ eigenvals @ eigenvecs


def grad_of_grad_var_compute(mean_grad, dd):
    '''
        DEPRECATED
        Does not store full Hessians in memory,
        but calculates mean Hessian from Hessians of each minibatch.
        Also scales poorly.

        LG[i] = 2 * (∇^2 L_k(Θ)[i] - ∇^2 L(Θ)[i]) @ (∇L_k(Θ) - ∇L(Θ))
                    (1,N)                     (N,1)

        For B total NxN matrix with m-rank eigenapproximation,
        Time:  O(B N^2 M^2)
        Space: O(BN)
    '''
    curr_time = datetime.datetime.now()
    N = mean_grad.shape[0]
    B = len(dd['eigenvectors'])

    learning_gradient = torch.zeros(mean_grad.shape)
    for i in range(N):
        hrows = torch.empty((B, N))
        for j, (eigenvecs, eigenvals) in enumerate(
                zip(dd['eigenvectors'], dd['eigenvalues'])):
            hrow = compute_row_from_eigendecomp(eigenvecs, eigenvals, i)
            hrows[j] = hrow
        mean_hrow = torch.mean(hrows, axis=0)

        for g, hrow in zip(dd['grad'], hrows):
            learning_gradient[i] += 2 * (hrow - mean_hrow) @ (g - mean_grad)

        t = datetime.datetime.now()
        diff = t - curr_time
        print(f'\t{i}/{N}: {diff}')
        curr_time = t


    return learning_gradient


'''
    Stats
'''
def get_gradients_over_minibatches(model, device, train_loader, optimizer, args,
        get_hessian = False):
    '''
        Process args['stats_samplesize'] minibatches, collecting gradients and optionally Hessians.
        Minibatches from the dataloader are not consumed by anything: we sample randomly from all minibatches in the current epoch.
    '''
    num_minibatches = args['stats_samplesize']
    num_eigens = args['num_eigens_hessian_approx']
    num_params = args['num_params']

    dd = {
        'loss': torch.empty((num_minibatches, 1)),
        'grad': torch.empty((num_minibatches, num_params)),
        'grad norm': torch.empty((num_minibatches, 1)),
    }
    if get_hessian:
        dd['eigenvalues'] = torch.empty((num_minibatches, num_eigens, num_eigens))
        dd['eigenvectors'] = torch.empty((num_minibatches, num_eigens, num_params))
        dd['mean eigenvalues'] = torch.empty((num_eigens, num_eigens))
        dd['mean eigenvectors'] = torch.empty((num_eigens, num_params))
        batches = []        

    for i, batch in enumerate(iter_sample_fast(train_loader, num_minibatches)):
        data, target = batch
        data, target = data.to(device), target.to(device)
        output = model(data)

        optimizer.zero_grad()
        loss = args['loss_func'](output, target)
        loss.backward()

        grad = torch.Tensor(gradient(model).cpu())
        dd['loss'][i] = torch.Tensor([loss.item()])
        dd['grad'][i] = grad
        dd['grad norm'][i] = gradient_norm(model)

        if get_hessian:
            eigenvals, eigenvecs = eigen_approx_hessian(model, [batch], args)
            dd['eigenvalues'][i]  = eigenvals
            dd['eigenvectors'][i] = eigenvecs
            batches.append(batch)

    if get_hessian:
        mean_eigenvals, mean_eigenvecs = eigen_approx_hessian(model,
                batches, args)
        dd['mean_eigenvalues'] = mean_eigenvals
        dd['mean_eigenvectors'] = mean_eigenvecs

    return dd


def optimize_grad_var(model, device, train_loader, optimizer, args):
    '''
        Calculate the gradient of the variance of gradients for NLL loss function.

        Var(∇L(Θ)) = E_k[|∇L_k(Θ) - ∇L(Θ)|^2]

        ∇ Var(∇L(Θ)) = E_k[
            2 * (∇^2 L_k(Θ) - ∇^2 L(Θ)) * (∇L_k(Θ) - ∇L(Θ))
        ]

        which has shape (N,1) when ∇L(Θ) has shape (N,1). 

        Uses low-rank approximation to Hessian, calculated on the fly (avoid storing whole matrix in memory).
    '''
    dd = get_gradients_over_minibatches(model, device, train_loader,
            optimizer, args, get_hessian=True)
    grad_var, mean_grad = gradient_variance(dd['grad'])
    
    learning_gradient = grad_of_grad_var(mean_grad, dd)

    losses = np.array(dd['loss'])
    grad_norms = np.array(dd['grad norm'])
    stats_d = {
        'loss_mean': np.mean(losses), 
        'loss_var': np.var(losses), 
        'gradnorm_mean': np.mean(grad_norms),
        'gradnorm_var': np.var(grad_norms),
        'Grad var': grad_var,
    }
    return stats_d, learning_gradient


def optimize_grad_squared_norm(model, device, batch, optimizer, args):
    '''
        Optimize the squared norm of the gradient of NLL loss.

            ∇(∇L(Θ)^2) = 2 * ∇L(Θ)^T * ∇^2 L(Θ)

        which has shape (N,1) when ∇L(Θ) has shape (N,1). 
    '''
    data, target = batch
    data, target = data.to(device), target.to(device)

    output = model(data)

    optimizer.zero_grad()
    loss_val = args['loss_func'](output, target)
    loss_val.backward()

    grad_squared_norm = gradient_norm(model)

    stats_d = {
        'Loss val': loss_val,
        'Grad squared norm': grad_squared_norm,
    }

    # Compute gradient on grad squared norm for learning
    g = gradient(model)
    eigenvals, eigenvecs = eigen_approx_hessian(model,
        batch, args)
    approx_hessian = eigenvecs.T @ np.diag(eigenvals) @ eigenvecs

    learning_gradient = 2 * approx_hessian @ np.array(g)
    return stats_d, learning_gradient


def calc_loss_var(model, device, train_loader, args):
    '''
      Calculate variance of loss for NLL loss function.
      Can be used as loss: Backprop-able
    '''
    dd = defaultdict(list)
    for data, target in iter_sample_fast(train_loader, args['stats_samplesize']):
        data, target = data.to(device), target.to(device)
        output = model(data)

        loss = args['loss_func'](output, target)
        dd['Loss'].append(loss)
    
    losses = torch.stack(dd['Loss'])
    return torch.var(losses, 0)


def get_stats(model, device, train_loader, optimizer, args):
    '''
      Compute across minibatches and gather stats on:
        Loss, Gradient norm, Gradient
      Randomly samples batches from `train_loader`.
    '''
    dd = get_gradients_over_minibatches(model, device, train_loader,
            optimizer, args)

    losses = np.array(dd['loss'])
    grad_norms = np.array(dd['grad norm'])
    grad_var, mean_grad = gradient_variance(dd['grad'])

    stats_d = {
        'loss_mean': np.mean(losses), 
        'loss_var': np.var(losses), 
        'gradnorm_mean': np.mean(grad_norms),
        'gradnorm_var': np.var(grad_norms),
        'grad_var': grad_var.item(),
    }
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
        raise ValueError(f"Sample larger than population. Got {len(results)} when {samplesize} was requested.\n")
    random.shuffle(results)  # Randomize their positions
    for i, v in enumerate(iterator, samplesize):
        r = random.randint(0, i)
        if r < samplesize:
            results[r] = v  # at a decreasing rate, replace random items
    return results


def write_args(args):
    os.makedirs(args['results_dir'])
    with open(args['results_dir'] + '/args.txt', 'w') as f:
        f.write(str(datetime.datetime.now()) + '\n')
        for k, v in args.items():
            f.write(f'{k}: {v}\n')
    return