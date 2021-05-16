import torch
import datetime

curr_time = datetime.datetime.now()

args = {
    'batch_size': 64,
    'test_batch_size': 1000,

    'epochs': 200,

    'lr': 0.1,
    'gamma': 0.7,

    'num_epochs_per_save': 1,
    'shuffle_batches_every_epoch': True,

    'seed': 1,
    'log_interval': 10,
    'use_cuda': torch.cuda.is_available(),
    'dry_run': False,

    'data_dir': '../data/',
    'results_dir': '../out/' + str(curr_time).replace(' ', '_') + '/',

    # function takes (output, target). Expected to be overwritten.
    'loss_func': None,
    'learning_func_name': 'standard',
    # [standard, grad_var, loss_var]

    'stats_samplesize': 5,
    'num_eigens_hessian_approx': 1,
}