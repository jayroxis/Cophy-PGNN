## Results

Using the default parameters in *./scripts/paramters.py*, the result files will be automatically generated in and saved to this directory.

Then, the result *.txt* file can be processed by *./scripts/parse_results.py*.

Apart from results text file, the detailed losses will also be recored in the folder in *.csv* format.

Here is an example of a result file named *result_02-08-2020-07:13:14_-0x40024e94a08a3052.txt*. The naming includes timestamp when the task is processed and its unique hash identifier.
~~~~
{'name': 'S-DNN', 'model_hash': '0x7f15d67b67f', 'model_path': '../models/-0x40024e94a08a3052.pt', 'log_path': '../logs/log_-0x40024e94a08a3052.txt', 'loss_path': '../logs/loss_-0x40024e94a08a3052.txt', 'loss_csv': '../results/-0x40024e94a08a3052.csv', 'mse_all': 0.03417858108878136, 'mse_wave_funtion': 0.02257717214524746, 'mse_energy': 0.2198011428117752, 'wave_function_overlap': 0.9231748580932617, 'val_mse_all': 7.377555448329076e-05, 'val_mse_wave_funtion': 7.827288209227845e-05, 'val_mse_energy': 1.8182681742473505e-06, 'val_wave_function_overlap': 0.9997082948684692, 'train_mse_all': 7.342341996263713e-05, 'train_mse_wave_funtion': 7.789656228851527e-05, 'train_mse_energy': 1.853163325904461e-06, 'train_wave_function_overlap': 0.9997082948684692, 'param_hash': '-0x79bf0193e5fe33ce', 'params': {'name': 'S-DNN', 'nn_params': {'hidden_size': 100, 'depth': 3, 'activation': <class 'torch.nn.modules.activation.Tanh'>, 'device': device(type='cuda', index=0), 'in_dim': 44, 'out_dim': 17}, 'train_params': {'epochs': 500, 'train_loss': ['mse_loss', 'phy_loss'], 'test_loss': [], 'num_batch': 1, 'optimizer': <class 'torch.optim.adamax.Adamax'>, 'cyclical': {}, 'L2_reg': 0.0, 'verbose': False, 'print_interval': 10, 'early_stopping': {'patience': 50, 'verbose': False, 'delta': 0, 'saved_model': '../models/-0x40024e94a08a3052.pt'}, 'loss': {'lambda_s': 9.361225617557782, 'lambda_e0': 1.2306341797931508, 'anneal_factor': 0.7141806599064502, 'anneal_interval': 10, 'norm_wf': False, 'noise': {}, 'cyclical': {}, 'train_loss': ['mse_loss', 'phy_loss'], 'test_loss': []}, 'data': <data_loader.DatasetLoader object at 0x7f163580dc50>, 'model': DNN(
  (layers): Sequential(
    (input): Linear(in_features=44, out_features=100, bias=True)
    (input_activation): Tanh()
    (hidden_0): Linear(in_features=100, out_features=100, bias=True)
    (activation_0): Tanh()
    (hidden_1): Linear(in_features=100, out_features=100, bias=True)
    (activation_1): Tanh()
    (hidden_2): Linear(in_features=100, out_features=100, bias=True)
    (activation_2): Tanh()
    (output): Linear(in_features=100, out_features=17, bias=True)
  )
)}, 'io_params': {'path_out': '../results/', 'path_fig': '../figures/', 'path_log': '../logs/', 'env_path': [], 'use_timestamp': True}, 'data_params': {'data_path': '//home/elhamod/melhamodenv/CMT/downloadedData/Ising/', 'phase': 'single-phase', 'n_sites': 4, 'train_size': 20000, 'val_size': 2000, 'test_size': 0, 'normalize_input': True, 'normalize_output': False, 'device': device(type='cuda', index=0), 'dataset': 'new'}, 'loss_params': {'lambda_s': 9.361225617557782, 'lambda_e0': 1.2306341797931508, 'anneal_factor': 0.7141806599064502, 'anneal_interval': 10, 'norm_wf': False, 'noise': {}, 'cyclical': {}, 'train_loss': ['mse_loss', 'phy_loss'], 'test_loss': []}}, 'epoch': 499, 'train_time': 91.83835172653198, 'time_per_epoch': 0.18404479303914226}
~~~~

After parsing by the *./scripts/parse_results.py*, the file looks like this:
~~~~ python
>>> import parse_results
>>> import json
>>> d = parse_results.parse_results([path_to_file] + "results/result_02-08-2020-07:13:14_-0x40024e94a08a3052.txt")
>>> print(json.dumps(d, indent=4))
~~~~

~~~~ json
out [4]:
{
    "name": "S-DNN",
    "model_hash": "0x7f15d67b67f",
    "model_path": "../models/-0x40024e94a08a3052.pt",
    "log_path": "../logs/log_-0x40024e94a08a3052.txt",
    "loss_path": "../logs/loss_-0x40024e94a08a3052.txt",
    "loss_csv": "../results/losses_-0x40024e94a08a3052.csv",
    "mse_all": 0.03417858108878136,
    "mse_wave_funtion": 0.02257717214524746,
    "mse_energy": 0.2198011428117752,
    "wave_function_overlap": 0.9231748580932617,
    "val_mse_all": 7.377555448329076e-05,
    "val_mse_wave_funtion": 7.827288209227845e-05,
    "val_mse_energy": 1.8182681742473505e-06,
    "val_wave_function_overlap": 0.9997082948684692,
    "train_mse_all": 7.342341996263713e-05,
    "train_mse_wave_funtion": 7.789656228851527e-05,
    "train_mse_energy": 1.853163325904461e-06,
    "train_wave_function_overlap": 0.9997082948684692,
    "param_hash": "-0x79bf0193e5fe33ce",
    "params": {
        "name": "S-DNN",
        "nn_params": {
            "hidden_size": 100,
            "depth": 3,
            "activation": "<class'torch.nn.modules.activation.Tanh'>",
            "device": "device(type='cuda',index=0)",
            "in_dim": 44,
            "out_dim": 17
        },
        "train_params": {
            "epochs": 500,
            "train_loss": [
                "mse_loss",
                "phy_loss"
            ],
            "test_loss": [],
            "num_batch": 1,
            "optimizer": "<class'torch.optim.adamax.Adamax'>",
            "cyclical": {},
            "L2_reg": 0.0,
            "verbose": false,
            "print_interval": 10,
            "early_stopping": {
                "patience": 50,
                "verbose": false,
                "delta": 0,
                "saved_model": "../models/-0x40024e94a08a3052.pt"
            },
            "loss": {
                "lambda_s": 9.361225617557782,
                "lambda_e0": 1.2306341797931508,
                "anneal_factor": 0.7141806599064502,
                "anneal_interval": 10,
                "norm_wf": false,
                "noise": {},
                "cyclical": {},
                "train_loss": [
                    "mse_loss",
                    "phy_loss"
                ],
                "test_loss": []
            },
            "data": "<data_loader.DatasetLoaderobjectat0x7f163580dc50>",
            "model": "DNN((layers):Sequential((input):Linear(in_features=44,out_features=100,bias=True)(input_activation):Tanh()(hidden_0):Linear(in_features=100,out_features=100,bias=True)(activation_0):Tanh()(hidden_1):Linear(in_features=100,out_features=100,bias=True)(activation_1):Tanh()(hidden_2):Linear(in_features=100,out_features=100,bias=True)(activation_2):Tanh()(output):Linear(in_features=100,out_features=17,bias=True)))"
        },
        "io_params": {
            "path_out": "../results/",
            "path_fig": "../figures/",
            "path_log": "../logs/",
            "env_path": [],
            "use_timestamp": true
        },
        "data_params": {
            "data_path": "//home/elhamod/melhamodenv/CMT/downloadedData/Ising/",
            "phase": "single-phase",
            "n_sites": 4,
            "train_size": 20000,
            "val_size": 2000,
            "test_size": 0,
            "normalize_input": true,
            "normalize_output": false,
            "device": "device(type='cuda',index=0)",
            "dataset": "new"
        },
        "loss_params": {
            "lambda_s": 9.361225617557782,
            "lambda_e0": 1.2306341797931508,
            "anneal_factor": 0.7141806599064502,
            "anneal_interval": 10,
            "norm_wf": false,
            "noise": {},
            "cyclical": {},
            "train_loss": [
                "mse_loss",
                "phy_loss"
            ],
            "test_loss": []
        }
    },
    "epoch": 499,
    "train_time": 91.83835172653198,
    "time_per_epoch": 0.18404479303914226
}
~~~~

A worth mentioned point is the other related file can be found in the result file. For example, where are the corresponding **model** and **logs**, **loss** files saved.
