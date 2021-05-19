'''
    Toy version of colored mnist.
    data = [binary color] + [noisy indicator variable for 10 digits]
'''
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import Dataset, DataLoader

import os, random, argparse
import numpy as np, pandas as pd

import lib, net, arguments

params = {
    ## Model
    # 'x_dim': 11,
    # 'width': 8,
    'width': 32,
    # 'width': 2048,
    'y_dim': 2,

    ## Learning
    # 'batch_size': 16,
    # 'batch_size': 64,
    'batch_size': 128,
    # 'batch_size': 1000,
    # 'batch_size': 1024,
    # 'batch_size': 2048,
    # 'batch_size': 5012,
    # 'batch_size': 10000,

    'lr': 1e-1,
    # 'lr': 1e0,
    # 'lr': 1e1,
    # 'lr': 1e2,
    # 'lr': 1e3,
    # 'lr': 1e4,

    # 'epochs': 10,
    # 'epochs': 20,
    'epochs': 500,

    'test_batch_size': 1000,

    'grad_var_weight': 0,
    # 'grad_var_weight': 1e-10,
    # 'grad_var_weight': 0.1,
    # 'grad_var_weight': 10,

    'loss_var_weight': 0, 
    # 'loss_var_weight': 10, 
    # 'loss_var_weight': 100, 
    # 'loss_var_weight': 1000, 
    # 'loss_var_weight': 3000, 

    # 'grad_norm_weight': 0,
    # 'grad_norm_weight': 0.5,
    # 'grad_norm_weight': 0.7,
    # 'grad_norm_weight': 0.80,
    # 'grad_norm_weight': 0.85,
    # 'grad_norm_weight': 1,
    # 'grad_norm_weight': 1.3,
    # 'grad_norm_weight': 1.5,
    # 'grad_norm_weight': 1.6,
    # 'grad_norm_weight': 2.5,
    # 'grad_norm_weight': 4.0,
    # 'grad_norm_weight': 5.0,
    # 'grad_norm_weight': 6.0,
    'grad_norm_weight': 7.0,
    # 'grad_norm_weight': 10,
    # 'grad_norm_weight': 12,
    # 'grad_norm_weight': 20,

    # 'L2_weight': 0,
    'L2_weight': 1e-6,
    # 'L2_weight': 0.018,

    'stats_samplesize': 20,
    'num_eigens_hessian_approx': 1,

    ## Data
    'N': 1000,
    'num_digits': 10,

    # Real CMNIST
    'noise': 0.05,
    'label_noise': 0.25,
    'train_color_class_prob': 0.85,
    'test_color_class_prob': 0.10,

    # Mine
    # 'noise': 0.1,
    # 'label_noise': 0.10,
    # 'train_color_class_prob': 0.80,
    # 'test_color_class_prob': 0.20,

    ## Misc
    'log_interval': 1000,
    # 'seed': 3,
    'warmup': 0,
    'plateau_patience': 3,
}

params['num_batches_per_epoch'] = params['N'] // params['batch_size']

params['stats_samplesize'] = min(params['stats_samplesize'],
                                 params['num_batches_per_epoch'])

params['x_dim'] = params['num_digits'] + 1

'''
    Special stats
'''
def get_loss_var_across_colors(args, model, train_data):
    model.eval()
    batch_size = 16
    num_reps = 50
    dd = defaultdict(list)
    for rep in range(num_reps):
        for num_color in range(batch_size):
            for group in [0, 1]:
                batch = get_custom_batch(train_data,
                    num_color, batch_size, group)
                data, target = batch

                output = model(data)
                loss = args['loss_func'](output, target)
                dd['Loss'].append(loss.item())

                correct = sum(torch.max(output, 1).indices == target).item()
                dd['Accuracy'].append(correct / len(target))

                dd['Num exp color'].append(num_color)
                dd['Frac exp color'].append(num_color / batch_size)
                dd['Batch size'].append(batch_size)
                dd['Replicate'].append(rep)

    fn = f'toy_{params["batch_size"]}batch'
    df = pd.DataFrame(dd)

    fna = fn + f'_gn{params["grad_norm_weight"]}'
    dfa = df.groupby('Frac exp color').agg('mean')

    var = np.var(dfa['Loss'])
    print(f'Loss var: {var:.6f}')

    df.to_csv(fn + '.csv')
    dfa.to_csv(fna + '.csv')
    return dd


'''
    Training
'''
def train_special(args, model, device, train_loader, optimizer, epoch):
    model.train()
    last_lr = optimizer.param_groups[0]['lr']
    
    stats = {}
    total_loss = []
    outputs = defaultdict(list)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = args['loss_func'](output, target)
        loss_val = loss.item()

        if params['num_batches_per_epoch'] > 1 and epoch > params['warmup']:
            if params['grad_var_weight'] > 0:
                stats, grad = lib.optimize_grad_var(model,
                    device, train_loader, optimizer, args)
                grad *= params['grad_var_weight']
                lib.assign_gradient_to_model(model, grad, device)
                loss_val += params['grad_var_weight'] * stats['grad_var']

            if params['loss_var_weight'] > 0:
                loss_var = lib.calc_loss_var(model, device, train_loader, args)
                loss += params['loss_var_weight'] * loss_var
                loss_val += params['loss_var_weight'] * loss_var.item()

            if params['grad_norm_weight'] > 0:
                stats, grad = lib.optimize_grad_squared_norm(model,
                    device, (data, target), optimizer, args)
                weight = params['grad_norm_weight']
                grad *= weight
                lib.assign_gradient_to_model(model, grad, device)
                loss_val += weight * stats['Grad squared norm']

        # Must backward after optional regularizers, since optimizing grad var/loss var code calls optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss_val)

        if batch_idx % args['log_interval'] == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]')

            correct = sum(torch.max(output, 1).indices == target).item()
            stats['Accuracy'] = correct / len(target)

            for k, v in stats.items():
                print(f'{k}: {v}')
            if args['dry_run']:
                break

            # stats_d = lib.get_stats(model, device, train_loader, optimizer, args)
            # for k, v in stats_d.items():
            #     print(f'{k}:\t{v}')

            # Save outputs to dictionary
            outputs["epoch"].append(epoch)
            outputs["batch_num"].append(batch_idx * len(data))
            # for stat in stats_d:
                # outputs[stat].append(stats_d[stat])

    # At the end of epoch save to file
    df = pd.DataFrame.from_dict(outputs)
    df.to_csv(os.path.join(args['results_dir'], f"outputs_train_epoch{epoch}.csv"))
    return np.mean(total_loss)


def valid(args, model, device, optimizer, data_loader):
    model.eval()
    valid_losses = []
    reg_losses = []
    correct = 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)

        valid_losses.append(args['loss_func'](output, target).item())

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if params['grad_norm_weight'] > 0:
            stats, grad = lib.optimize_grad_squared_norm(model,
                device, (data, target), optimizer, args)
            weight = params['grad_norm_weight']
            grad *= weight
            reg_losses.append(weight * stats['Grad squared norm'])


    print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        np.mean(valid_losses), correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    print(f'Regularizer loss: {np.mean(reg_losses)}\n')
    return np.mean(valid_losses), np.mean(reg_losses)


def test(args, model, device, test_loader):
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
    return test_loss


'''
    Main trainer
'''
def trainer(model, train_loader, valid_loader, test_loader, args):
    # Training settings
    args['num_params'] = sum([np.prod(p.size()) for p in model.parameters()])
    print(args)
    lib.write_args(args)

    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if args['use_cuda'] else "cpu")

    model.to(device)

    # optimizer = optim.Adadelta(model.parameters(), lr=args['lr'], 
    #         weight_decay=params['L2_weight'])
    optimizer = optim.SGD(model.parameters(), lr=args['lr'],
            weight_decay=params['L2_weight'])

    # scheduler = StepLR(optimizer, step_size=1, gamma=args['gamma'])
    scheduler = ReduceLROnPlateau(optimizer, patience=args['plateau_patience'], verbose=True)

    recorder = defaultdict(list)
    for epoch in range(1, args['epochs'] + 1):
        total_train_loss = train_special(args, model, device, train_loader, optimizer, epoch)
        train_loss = test(args, model, device, train_loader)
        # valid_loss, valid_reg = valid(args, model, device,
        #     optimizer, valid_loader)
        valid_loss = test(args, model, device, valid_loader)
        test_loss = test(args, model, device, test_loader)


        recorder['epoch'].append(epoch)
        recorder['total train loss'].append(total_train_loss)
        recorder['train loss'].append(train_loss)
        recorder['valid loss'].append(valid_loss)
        recorder['test loss'].append(test_loss)

        print('Total training loss:', total_train_loss)
        scheduler.step(total_train_loss)

        # total_valid_loss = valid_loss + valid_reg
        # recorder['total valid loss'].append(total_valid_loss)
        # scheduler.step(total_valid_loss)

        if epoch % args['num_epochs_per_save'] == 0:
            torch.save(model.state_dict(), os.path.join(args['results_dir'], f"model_epoch{epoch}.pt"))
            df = pd.DataFrame.from_dict(recorder)
            df.to_csv(os.path.join(args['results_dir'], f"outputs_test.csv"))

        converge_epochs = 30
        if epoch > converge_epochs:
            if np.median(recorder['valid loss'][-converge_epochs:]) / valid_loss < 1.0001:
                print('Detected convergence in validation loss')
                break

    print('Done.')
    return model


'''
    Data
'''
class ToyColoredMNIST(Dataset):
    def __init__(self, train = True):
        '''
            0-4: Label 0 with high probability
                In training data, color 0 with high probability
            0-4: Label 1 with high probability
                In training data, color 1 with high probability
        '''
        N = params['N']
        noise = params['noise']
        label_noise = params['label_noise']
        num_digits = params['num_digits']
        num_per_class = N // num_digits

        # 1 - => 80% color class prob = 80% time, color = label (0 = 0)
        if train:
            color_class_prob = 1 - params['train_color_class_prob']
        elif not train:
            color_class_prob = 1 - params['test_color_class_prob']

        dd = defaultdict(list)
        for i in range(num_digits):
            if i < num_digits // 2:
                y_probs = [1 - label_noise, label_noise]
            else: 
                y_probs = [label_noise, 1 - label_noise]

            y = np.random.choice([0.0, 1.0],
                    size=(num_per_class, 1), p=y_probs)

            color_vec = np.logical_xor(y,
                np.random.binomial(1, color_class_prob, size=y.shape))
            color_vec = color_vec.astype(float)

            x = np.random.normal(0, noise, (num_per_class, num_digits))
            x[:,i] += 1

            x = np.hstack([color_vec, x])
            dd['X'].append(torch.Tensor(x))
            dd['Y'].append(torch.Tensor(y))
        
        self.X = torch.cat( dd['X'] )
        self.Y = torch.cat( dd['Y'] ).squeeze().long()
        self.dd = dd

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def get_custom_batch(dataset, num_exp_color, batch_size, group):
    '''
        Focuses on lower numbers only, so that
        expected color = 0 in training data

        When num_exp_color is 0, return a batch
        with all unexpected colors (e.g., color 1 for lower numbers)
    '''
    if group == 0:
        x1, y1 = get_data_of_color(dataset, 0, num_exp_color, group)
        x2, y2 = get_data_of_color(dataset, 1, batch_size - num_exp_color, group)
    elif group == 1:
        x1, y1 = get_data_of_color(dataset, 1, num_exp_color, group)
        x2, y2 = get_data_of_color(dataset, 0, batch_size - num_exp_color, group)
    return torch.stack(x1 + x2), torch.stack(y1 + y2).squeeze().long()


def get_data_of_color(dataset, color, num_wanted, group):
    '''
        Finds `num_wanted` datapoints with specified color.
    '''
    found_xs, found_ys = [], []
    xs = dataset.dd['X']
    ys = dataset.dd['Y']

    nd = params['num_digits']
    nums = list(range(nd))
    if group == 0:
        nums = nums[:nd//2]
    else:
        nums = nums[nd//2:]

    get_color = lambda x: x[0]
    for _ in range(num_wanted):
        num = np.random.choice(nums)

        i = np.random.randint(0, len(xs[num]))
        x = xs[num][i]
        while get_color(x) != color:
            i = np.random.randint(0, len(xs[num]))
            x = xs[num][i]
        # y = ys[num][i]
        y = torch.Tensor([float(group)])

        found_xs.append(x)
        found_ys.append(y)
    return found_xs, found_ys

'''
    Main
'''
def main():
    args = arguments.args
    args.update(params)

    args['loss_func'] = F.cross_entropy

    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])

    model = net.MLP(params['x_dim'], params['width'], params['y_dim'])
    # model = net.Linear(params['x_dim'], params['y_dim'])

    train_data = ToyColoredMNIST(train=True)
    valid_data = ToyColoredMNIST(train=True)
    test_data = ToyColoredMNIST(train=False)

    train_loader = DataLoader(train_data,
        batch_size=params['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data,
        batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data,
        batch_size=params['test_batch_size'], shuffle=True)

    model = trainer(model, train_loader, valid_loader, test_loader, args)

    new_data = ToyColoredMNIST(train=True)
    get_loss_var_across_colors(args, model, new_data)
    return


if __name__ == '__main__':
    main()
