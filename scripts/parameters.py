import torch
from torch import optim

import numpy as np

# ========================= The Parameter Class =========================
class Params(object):
    def __init__(
        self, 
        nn_params={}, 
        train_params={},
        io_params={},
        data_params={},
        loss_params={},
        name='default'
    ):
        self.name = name
        
        # default nerual network parameters
        self.nn_params = {
            'hidden_size': 100,
            'depth': 3,               # network depth = 'depth' + 1
            'activation': torch.nn.Tanh,
            'device': torch.device('cuda:0'),
            'softmax': False
        }
        
        # default training parameters
        self.train_params = {
            'epochs': 500,
            'train_loss': [
                'mse_loss', 
                'phy_loss', 
                'energy_loss'
            ],
            'test_loss': [
                'phy_loss', 
                'energy_loss'
            ],
            'num_batch': 1,
            'steplr': 200,
            'gamma': 0.5,
            'optimizer': torch.optim.Adamax,
            'lr': 0.002,
            'cyclical': {
                'base_lr': 0.001, 
                'max_lr': 0.1, 
                'step_size_up': 20, 
                'step_size_down': 20, 
                'mode': 'triangular',
                'cycle_momentum': False
            },     # set to False or {} to disable
            'L2_reg': 0.0,
            'verbose': False,
            'print_interval': 10,
            'early_stopping': False,
            'vanilla': False
        }
        
        # default input/output parameters
        self.io_params = {
            'path_out': '../results/',
            'path_fig': '../figures/',
            'path_log': '../logs/',
            'env_path': [],           # path that will be registered through sys.path
            'use_timestamp': True
        }
        
        # default dataset parameters
        self.data_params = {
            'data_path': '../datasets/',
            'phase': 'single-phase',  # use 'single-phase', 'dual-phase' or 'multi-phase'
            'n_sites': 4,
            'train_size': 20000,
            'val_size': 2000,
            'test_size': 0,           # set to 0 to use all the test data
            'normalize_input': True,
            'normalize_output': False,
            'device': torch.device('cuda:0'),
            'dataset': 'new'
        }
        
        # default loss function parameters
        self.loss_params = {
            'lambda_s': 1.0,
            'lambda_e0': 0.2,
            'anneal_factor': 0.9,
            'anneal_interval': 50,    # this is a parameter also for noise and cyclical update 
            'norm_wf': False,
             # set to False or None or {} to disable the noise
            'noise': {'mode': 'uniform', 'mean': 0.0, 'var': 0.5, 'decay': 0.9},
            'cyclical': {'mode': 'sin', 'decay': 0.9, 'mean': 1.0, 'amp': 0.5, 'period': 20},
            'cold_start': {}
        }

        # update params
        self.nn_params.update(nn_params)
        self.train_params.update(train_params)
        self.io_params.update(io_params)
        self.data_params.update(data_params)
        self.loss_params.update(loss_params)