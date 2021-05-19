from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class MLP(nn.Module):
    def __init__(self, inp_dim, width, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(inp_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = self.fc3(x)
        return output


class Linear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(inp_dim, out_dim)

    def forward(self, x):
        return self.fc1(x)


class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def load_pretrained_model(net_class, pt_file, args):
    model = net_class()
    model.load_state_dict(torch.load(pt_file))
    args['Model is pretrained'] = True
    args['Model class'] = str(net_class)
    args['Model pretrained file'] = pt_file
    return model