'''
'''

import torch
import data, trainer, net, arguments

args = arguments.args

def local_mnist():
    # laptop
    args['data_dir'] = '../data/'

    train_loader, test_loader = data.mnist(args)
    model = net.MNIST_Net()

    trainer.main(model, train_loader, test_loader, args)
    return


'''
    On gcloud or aws, overwrite:
    args['data_dir'], if data is already downloaded
    args['results_dir'], to save to long-term storage

    optionally, load pre-trained model 
'''


if __name__ == '__main__':
    local_mnist()