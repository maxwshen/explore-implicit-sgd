'''
'''
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import os, random, argparse
import numpy as np, pandas as pd

import lib, dataloader

'''
    Train/test
'''
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    if args['subset_batches']:
        # Running get_stats twice does not produce identical output. Happens due to dropout and stochastic layers. Switching to model.eval() fixes the problem. Preferred for investigating effect of optimizing grad/loss var on small subset of batches on pretrained model, to get cleaner stats.
        model.eval()

    stats = {}
    outputs = defaultdict(list)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = args['loss_func'](output, target)

        if args['learning_func_name'] == 'standard':
            loss.backward()
        elif args['learning_func_name'] == 'grad_var':
            stats, grad = lib.optimize_grad_var(model, device,
                    train_loader, optimizer, args)
            lib.assign_gradient_to_model(model, grad, device)
        elif args['learning_func_name'] == 'loss_var':
            loss_var = lib.calc_loss_var(model, device, train_loader, args)
            loss_var.backward()
            stats['Loss var'] = loss_var

        optimizer.step()

        if batch_idx % args['log_interval'] == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)] Loss standard: {loss.item():.3f}')
            for k, v in stats.items():
                print(f'{k}: {v}')
            if args['dry_run']:
                break

            stats_d = lib.get_stats(model, device, train_loader, optimizer, args)
            for k, v in stats_d.items():
                print(f'{k}:\t{v}')

            # Save outputs to dictionary
            outputs["epoch"].append(epoch)
            outputs["batch_num"].append(batch_idx * len(data))
            outputs["main loss"].append(loss.item())
            for stat in stats_d:
                outputs[stat].append(stats_d[stat])

    # At the end of epoch save to file
    df = pd.DataFrame.from_dict(outputs)
    df.to_csv(os.path.join(args['results_dir'], f"outputs_train_epoch{epoch}.csv"))
    return


def test(model, device, test_loader, args, loss_recorder_dict):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += args['loss_func'](output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    loss_recorder_dict['loss'].append(test_loss)
    return


'''
    Main trainer
'''
def main(model, train_loader, test_loader, args):
    # Training settings
    args['num_params'] = sum([np.prod(p.size()) for p in model.parameters()])
    print(args)
    lib.write_args(args)

    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if args['use_cuda'] else "cpu")

    model.to(device)

    # optimizer = optim.Adadelta(model.parameters(), lr=args['lr'])
    optimizer = optim.SGD(model.parameters(), lr=args['lr'])

    scheduler = StepLR(optimizer, step_size=1, gamma=args['gamma'])

    test_loss_recorder_dict = defaultdict(list)
    for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, args, test_loss_recorder_dict)
        test_loss_recorder_dict['epoch'].append(epoch)
        scheduler.step()

        if epoch % args['num_epochs_per_save'] == 0:
            torch.save(model.state_dict(), os.path.join(args['results_dir'], f"model_epoch{epoch}.pt"))
            df = pd.DataFrame.from_dict(test_loss_recorder_dict)
            df.to_csv(os.path.join(args['results_dir'], f"outputs_test.csv"))

    print('Done.')




