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

import lib


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    outputs = defaultdict(list)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        if args['loss_type'] == 'loss':
            loss = F.nll_loss(output, target)
        elif args['loss_type'] == 'grad_var':
            loss = lib.calc_grad_var(model, device, train_loader)
        elif args['loss_type'] == 'loss_var':
            loss = lib.calc_loss_var(model, device, train_loader)

        loss.backward()

        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            # grad_norm = lib.gradient_norm(model)
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]  Loss: {loss.item():.3f}  Grad norm: {grad_norm:.2f}')
            if args['dry_run']:
                break

            stats_d = lib.get_stats(model, device, train_loader)

            # Save outputs to dictionary
            outputs["epoch"].append(epoch)
            outputs["batch_num"].append(batch_idx * len(data))
            outputs["loss"].append(loss.item())
            # outputs["grad_norm"].append(grad_norm)
            for stat in stats_d:
                outputs[stat].append(stats_d[stat])

    # At the end of epoch save to file
    df = pd.DataFrame.from_dict(outputs)
    df.to_csv(os.path.join(args['results_dir'], f"outputs_train_epoch{epoch}.csv"))


def test(model, device, test_loader, loss_recorder_dict):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # Update the loss recorder
    loss_recorder_dict['loss'].append(test_loss)
    return loss_recorder_dict


def main(model, train_loader, test_loader, args):
    # Training settings
    print(args)

    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if args['use_cuda'] else "cpu")

    model.to(device)

    # optimizer = optim.Adadelta(model.parameters(), lr=args['lr'])
    optimizer = optim.SGD(model.parameters(), lr=args['lr'])

    scheduler = StepLR(optimizer, step_size=1, gamma=args['gamma'])

    test_loss_recorder_dict = defaultdict(list)
    for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_loss_recorder_dict = test(model, device, test_loader, test_loss_recorder_dict)
        test_loss_recorder_dict['epoch'].append(epoch)
        scheduler.step()

        if epoch % args['num_epochs_per_save'] == 0:
            torch.save(model.state_dict(), os.path.join(args['results_dir'], f"model_epoch{epoch}.pt"))
            df = pd.DataFrame.from_dict(test_loss_recorder_dict)
            df.to_csv(os.path.join(args['results_dir'], f"outputs_test.csv"))

    print('Done.')




