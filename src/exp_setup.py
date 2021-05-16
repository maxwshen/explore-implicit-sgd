'''
'''

import torch
import torch.nn.functional as F
import data, trainer, net, arguments, lib, dataloader

args = arguments.args

def local_mnist():
    # laptop
    args['data_dir'] = '../data/'
    args['loss_func'] = F.nll_loss

    train_loader, test_loader = data.mnist(args)
    model = net.MNIST_Net()

    trainer.main(model, train_loader, test_loader, args)
    return


def local_pretrained_mnist_lossvar():
    # laptop
    args['data_dir'] = '../data/'
    args['loss_func'] = F.nll_loss
    # args['learning_func_name'] = 'loss_var'
    args['learning_func_name'] = 'grad_var'
    args['stats_samplesize'] = 3
    args['num_eigens_hessian_approx'] = 1
    args['lr'] = 1e-3
    args['log_interval'] = 1

    train_loader, test_loader = data.mnist(args)

    batches = list(lib.iter_sample_fast(train_loader, args['stats_samplesize']))
    batch_loader = dataloader.get_subset_batch_loader(batches, args)
    args['subset_batches'] = True 
    print(f'\nTraining only on {args["stats_samplesize"]} batches of size {args["batch_size"]}!\n')

    pt_fn = '../data/models/mnist_model_epoch10.pt'
    model = net.load_pretrained_model(net.MNIST_Net, pt_fn, args)

    trainer.main(model, batch_loader, test_loader, args)
    return

'''
    On gcloud or aws, overwrite:
    args['data_dir'], if data is already downloaded
    args['results_dir'], to save to long-term storage

    optionally, load pre-trained model 
'''


if __name__ == '__main__':
    # local_mnist()
    local_pretrained_mnist_lossvar()
