'''
   Evaluate lib.optimize_grad_squared_norm
   TODO: Construct batches, simple model, loss function (need to make flexible)

   f(x) = x.T @ A @ x, for SPD matrix A. Maps (N,1) -> (1).
   Data: A
   Model: x.T @ A @ x, for 
   Learn x (convex function of loss)
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
    'y_dim': 1,

    'A_scale': 1,
    'area_lim': 10,

    'num_eigens_hessian_approx': 5,

    'lr': 1e-2,
}

'''
    Training
'''
def train_special(args, model, device, train_loader, optimizer, epoch):
    model.train()

    outputs = defaultdict(list)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        stats, grad = lib.optimize_grad_squared_norm(model,
            device, (data, target), args)
        lib.assign_gradient_to_model(model, grad)

        print(model(data))
        print(grad)

        optimizer.step()

        # print(model(data))
        # import code; code.interact(local=dict(globals(), **locals()))

        if batch_idx % args['log_interval'] == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]')
            for k, v in stats.items():
                print(f'{k}: {v}')
            if args['dry_run']:
                break

            stats_d = lib.get_stats(model,
                device, train_loader, args, samplesize=1)

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

    # optimizer = optim.Adadelta(model.parameters(), lr=args['lr'])
    optimizer = optim.SGD(model.parameters(), lr=args['lr'])

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
        d = params['x_dim']
        # self.optima = torch.Tensor(np.random.uniform(-L, L, d))
        self.optima = torch.zeros(d)
        self.A = torch.Tensor(make_spd_matrix(d) * params['A_scale'])
        # Making it negative definite confirms that we are minimizing the gradient norm, which maximizes the loss.
        self.A *= -1

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.Tensor([0.0]), torch.Tensor([0.0])

    def loss(self, x, target):
        x1 = torch.matmul((x - self.optima).T, self.A)
        val = torch.matmul(x1, (x - self.optima))
        return val


class PassThroughModel(nn.Module):
    def __init__(self):
        super(PassThroughModel, self).__init__()
        init = torch.rand(params['x_dim'], dtype=torch.float32)
        self.params = torch.nn.parameter.Parameter(init, requires_grad=True)

    def forward(self, x):
        return self.params


def main():
    args = arguments.args
    args.update(params)

    convex_data = ConvexDataset()

    args['loss_func'] = convex_data.loss

    model = PassThroughModel()

    train_loader = DataLoader(convex_data,
        batch_size=1, shuffle=True)

    trainer(model, train_loader, args)
    return


if __name__ == '__main__':
    main()