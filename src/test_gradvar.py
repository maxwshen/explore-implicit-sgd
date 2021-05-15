'''
   Evaluate lib.optimize_grad_var

   f(x) = x.T @ A @ x, for SPD matrix A. Maps (N,1) -> (1).
   Data: (x, f(x)) for random x
   Model: Two layer ReLU MLP.
'''
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

import os, random, argparse
import numpy as np, pandas as pd

import lib, net, arguments

params = {
    'x_dim': 5,
    'width': 8,
    'y_dim': 1,

    'N': 15,
    'batch_size': 5,
    'A_scale': 0.01,
    'noise': 0.01,
    'area_lim': 10,

    'num_eigens_hessian_approx': 1,
    'lr': 1e-1,

    # num minibatches. Must satisfy
    #   stats_samplesize * batch_size < N.
    'stats_samplesize': 3,
}

'''
    Training
'''
def train_special(args, model, device, train_loader, optimizer, epoch):
    model.train()
    
    stats = {}
    outputs = defaultdict(list)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if epoch <= 10:
            output = model(data)
            loss = args['loss_func'](output, target)
            loss.backward()
            print(f'Loss: {loss.item():.3f}')
        else:
            stats, grad = lib.optimize_grad_var(model,
                device, train_loader, optimizer, args)
            lib.assign_gradient_to_model(model, grad)

        optimizer.step()

        if batch_idx % args['log_interval'] == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]')
            for k, v in stats.items():
                print(f'{k}: {v}')
            if args['dry_run']:
                break

            stats_d = lib.get_stats(model, device, train_loader, optimizer, args)

            # Save outputs to dictionary
            outputs["epoch"].append(epoch)
            outputs["batch_num"].append(batch_idx * len(data))
            for stat in stats_d:
                outputs[stat].append(stats_d[stat])

    # At the end of epoch save to file
    df = pd.DataFrame.from_dict(outputs)
    df.to_csv(os.path.join(args['results_dir'], f"outputs_train_epoch{epoch}.csv"))


'''
    Main trainer
'''
def trainer(model, train_loader, args):
    # Training settings
    print(args)
    lib.write_args(args)

    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if args['use_cuda'] else "cpu")

    model.to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args['lr'])
    # optimizer = optim.SGD(model.parameters(), lr=args['lr'])

    scheduler = StepLR(optimizer, step_size=1, gamma=args['gamma'])

    for epoch in range(1, args['epochs'] + 1):
        train_special(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()

        if epoch % args['num_epochs_per_save'] == 0:
            torch.save(model.state_dict(), os.path.join(args['results_dir'], f"model_epoch{epoch}.pt"))

    print('Done.')


'''
    Data
'''
from sklearn.datasets import make_spd_matrix
class ConvexDataset(Dataset):
    def __init__(self):
        L = params['area_lim']
        N = params['N']
        d = params['x_dim']

        A = make_spd_matrix(d) * params['A_scale']
        dd = defaultdict(list)
        for i in range(N):
            x = np.random.uniform(-1 * L, L, d)
            xs = x.reshape(1, d)
            y = xs @ A @ xs.T + np.random.normal(0, params['noise'])
            dd['x'].append(x)
            dd['y'].append(y)

        self.X = torch.Tensor( np.array(dd['x']) )
        self.Y = torch.Tensor( np.array(dd['y']) ).squeeze(-1)
        self.A = A

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.Tensor(self.X[idx]), torch.Tensor(self.Y[idx])


def main():
    args = arguments.args
    args.update(params)

    args['loss_func'] = F.mse_loss

    model = net.MLP(params['x_dim'], 
        params['width'], 
        params['y_dim'])

    convex_data = ConvexDataset()
    train_loader = DataLoader(convex_data,
        batch_size=params['batch_size'], shuffle=False)

    trainer(model, train_loader, args)
    return


if __name__ == '__main__':
    main()