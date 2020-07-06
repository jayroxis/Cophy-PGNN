# Scripts

All the scripts are written under *Python 3.6*. The *training.py* defines **Trainer** class, which is the main entry for experiments. The **Trainer** constructor takes **Params** object defined in the *parameters.py*. Example of usage are in the *./notebooks* folder.

--------------------------------------------------

The default parameters are:

### default nerual network parameters
~~~~ python
self.nn_params = {
    'hidden_size': 100,
    'depth': 3,               # network depth = 'depth' + 1
    'activation': torch.nn.Tanh,
    'device': torch.device('cuda:0')
}
~~~~

### default training parameters
~~~~ python
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
    'optimizer': torch.optim.Adamax,
    'cyclical': {
        'base_lr': 0.001, 
        'max_lr': 0.1, 
        'step_size_up': 20, 
        'step_size_down': 20, 
        'mode': 'triangular'
    },     # set to False or {} to disable
    'L2_reg': 0.0,
    'verbose': False,
    'print_interval': 10,
    'early_stopping': False
}
~~~~

### default input/output parameters
~~~~ python
self.io_params = {
    'path_out': '../results/',
    'path_fig': '../figures/',
    'path_log': '../logs/',
    'env_path': [],           # path that will be registered through sys.path
    'use_timestamp': True
}
~~~~

### default dataset parameters
~~~~ python
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
~~~~

### default loss function parameters
~~~~ python
self.loss_params = {
    'lambda_s': 1.0,
    'lambda_e0': 0.2,
    'anneal_factor': 0.9,
    'anneal_interval': 50,    # this is a parameter also for noise and cyclical update 
    'norm_wf': False,
    # set to False or None or {} to disable the noise
    'noise': {'mode': 'uniform', 'mean': 0.0, 'var': 0.5, 'decay': 0.9}, 
    'cyclical': {'mode': 'sin', 'decay': 0.9, 'mean': 1.0, 'amp': 1.0, 'period': 20, 'cycle_momentum': False}
}
~~~~
