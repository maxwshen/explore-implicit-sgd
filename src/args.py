import torch
import datetime

args = {
  'batch_size': 64,
  'test_batch_size': 1000,

  'epochs': 200,

  'loss_type': 'loss_var',

  'lr': 0.1,
  'gamma': 0.7,

  'num_epochs_per_save': 1,

  'seed': 1,
  'log_interval': 10,
  'no_cuda': not torch.cuda.is_available(),
  'dry_run': False,

  'results_dir': str(datetime.datetime.now()).replace(' ', '_'),
}