# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# pytorch
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

# sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# scipy
from scipy import signal

# others
import json
import math
import time
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from fastprogress.fastprogress import master_bar, progress_bar

# disable warnings
import warnings
warnings.filterwarnings('ignore')

# seaborn aesthetic settings
sns.set_style("white")

from distutils.spawn import find_executable

import sys

# ====================== self defined scripts below ===================

# overwrite the text rendering using LaTex
if find_executable('latex'):
    from config_plots import global_settings
    global_settings()

# data loader
from data_loader import DatasetLoader

# the deep neural networks
from DNN import DNN

# all loss functions        
from loss_functions import *

# early stopping
from early_stopping import EarlyStopping

# parameters
from parameters import Params
        
# ========================= The Trainer Class =========================
class Trainer(object):
    def __init__(self, master_bar=None, plot=True):
        self.current_hash = ""
        self.master_bar = master_bar
        self.plot_flag = plot
        
    def start(self, param, verbose=False):
        current_str = self.str_now() + str(param.__dict__)
        self.current_hash = hex(hash(current_str))
        data_params, nn_params, train_params = dict(), dict(), dict()
        for key, value in param.data_params.items():
            data_params[key] = str(value)
        for key, value in param.nn_params.items():
            nn_params[key] = str(value)
        for key, value in param.train_params.items():
            train_params[key] = str(value)
        statistics = \
        """
==========================================================================
Action: initialize task.
Time: %s
Task Id: %s

io_params:
%s
data_params:
%s
nn_params:
%s
loss_params:
%s
train_params:
%s

    Status: Successful
--------------------------------------------------------------------------

        """ % \
        (
            self.str_now(),
            self.current_hash,
            json.dumps(param.io_params, indent=4),
            json.dumps(data_params, indent=4),
            json.dumps(nn_params, indent=4),
            json.dumps(param.loss_params, indent=4),
            json.dumps(train_params, indent=4)
        )
        if verbose:
            print(statistics)
            
        # load data    
        self.configure_io(param.io_params)
        
        # write log file
        log_file = open(
            self.path_log + 'log_' + self.current_hash + '.txt', 
            "a"
        )
        log_file.write(statistics)
        log_file.close()
        
        data = self.load_data(param.data_params)
        if verbose:
            print('Data loaded.')
        
        # build network
        nn_param = param.nn_params
        nn_param['in_dim'] = data.X_train_tensor.size(1)
        nn_param['out_dim'] = data.y_train_tensor.size(1)
        model = self.build_model(nn_param)
        if verbose:
            print('Model built.')
        
        # training
        train_param = param.train_params
        train_param['loss'] = param.loss_params
        train_param['data'] = data
        train_param['model'] = model
        model, stats = self.train(train_param)
        if verbose:
            print('Training Complete.')
            
        # evaluation
        self.evaluate(model, data, stats, param, name=param.name)
            
    def configure_io(self, io_param):
        # add system environment paths
        if 'env_path' in io_param:
            for path in io_param['env_path']:
                sys.path.append(path)
        self.path_out = io_param['path_out']
        self.path_fig = io_param['path_fig']
        self.path_log = io_param['path_log']
        
    def load_data(self, data_param, verbose=False):
        # load data
        if data_param['dataset'] == 'old':
            data_path = data_param['data_path'] + data_param['phase'] + '/'
        else:
            data_path = data_param['data_path']
        data = DatasetLoader(
            data_path=data_path, 
            n_sites=data_param['n_sites'],
            train_size=data_param['train_size'], 
            val_size=data_param['val_size'], 
            test_size=data_param['test_size']
        )
        
        # normalization
        data.normalization(x=data_param['normalize_input'], y=data_param['normalize_output'])
        
        # convert to torch tensor
        data.torch_tensor(device=data_param['device'])
        
        # print out
        statistics = \
        """
==========================================================================
Action: load data.
Time: %s
Task Id: %s

Training Inputs:    %s
Training Outputs:   %s
Validation Inputs:  %s
Validation Outputs: %s
Test Inputs:  %s
Test Outputs: %s

X Scaler: %s
Y Scaler: %s
Device:   %s
        Status: Successful
--------------------------------------------------------------------------
            
        """ % \
        (
            self.str_now(),
            self.current_hash,
            data.X_train.shape,
            data.y_train.shape,
            data.X_val.shape,
            data.y_val.shape,
            data.X_test.shape,
            data.y_test.shape,
            data.std_scaler_x,
            data.std_scaler_y,
            data_param['device']
        )
        
        if verbose:
            print(statistics)
        
        # write log file
        log_file = open(
                self.path_log + 'log_' + self.current_hash + '.txt', 
                "a"
            )
        log_file.write(statistics)
        log_file.close()
        
        return data
    
    def build_model(self, nn_param, verbose=False):
        # Compile model
        d_in = nn_param['in_dim']
        d_out = nn_param['out_dim']
        depth = nn_param['depth']
        h = nn_param['hidden_size']
        activation = nn_param['activation']
        device = nn_param['device']
        
        # parallelize if possible
        model = DNN(d_in, h, d_out, depth, act=activation)
#         if torch.cuda.is_available():
#             if torch.cuda.device_count() > 4:
#                 model = torch.nn.DataParallel(model)
        model.to(device)
        
        # print out
        statistics = \
        """
==========================================================================
Action: build model.
Time: %s
Task Id: %s

Input Dimension:    %d
Output Dimension:   %d
Model Depth:        %d
Hidden State Width: %d

Activation:  %s
Device:      %s

        Status: Successful
--------------------------------------------------------------------------
            
        """ % \
        (
            self.str_now(),
            self.current_hash,
            d_in,
            d_out,
            depth,
            h,
            activation,
            device,
        )
        
        if verbose:
            print(statistics)
        
        # write log file
        log_file = open(
            self.path_log + 'log_' + self.current_hash + '.txt', 
            "a"
        )
        log_file.write(statistics)
        log_file.close()
        return model
    
    def train(self, train_param, verbose=False):
        # unpack
        model = train_param['model']
        data = train_param['data']
        loss_param = train_param['loss']
        loss_param['train_loss'] = train_param['train_loss']
        loss_param['test_loss'] = train_param['test_loss']
        
        # write loss file
        loss_file = open(
            self.path_log + 'loss_' + self.current_hash + '.txt', 
            "a"
        )
        loss_file.write(
            '\n==========================================================================\n'
        )
        
        # get training data loader
        train_batch_size = math.ceil(data.y_train_tensor.size(0) / train_param['num_batch'])
        data_train_loader = DataLoader(
            list(zip(data.X_train_tensor, data.y_train_tensor, data.X_train_origin)), 
            batch_size=train_batch_size, 
            shuffle=True
        )
        
        # get test data loader
        val_batch_size = math.ceil(data.y_val_tensor.size(0) / train_param['num_batch'])
        data_val_loader = DataLoader(
            list(zip(data.X_val_tensor, data.y_val_tensor, data.X_val_origin)), 
            batch_size=val_batch_size, 
            shuffle=True
        )
        
        # get test data loader
        test_batch_size = math.ceil(data.y_test_tensor.size(0) / train_param['num_batch'])
        data_test_loader = DataLoader(
            list(zip(data.X_test_tensor, data.y_test_tensor, data.X_test_origin)), 
            batch_size=test_batch_size, 
            shuffle=True
        )
        
        # Optimizer
        optimizer = train_param['optimizer'](
            model.parameters(), 
            weight_decay=train_param['L2_reg']
        )
        
        # cyclical scheduler
        if train_param['cyclical']:
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **train_param['cyclical'])
        
        # MSE Loss
        criterion = torch.nn.MSELoss()
        
        # path to save model
        path_to_model = '../models/%s.pt' % self.current_hash
        
        # for early stopping
        if train_param['early_stopping']:
            train_param['early_stopping']['saved_model'] = path_to_model
            early_stopping = EarlyStopping(**train_param['early_stopping'])
            
        # For Recording Losses
        NUMEPOCHS = train_param['epochs']
        train_losses = np.zeros(NUMEPOCHS)
        val_losses = np.zeros(NUMEPOCHS)
        test_losses = np.zeros(NUMEPOCHS)
        train_phy_losses = np.zeros(NUMEPOCHS)
        val_phy_losses = np.zeros(NUMEPOCHS)
        test_phy_losses = np.zeros(NUMEPOCHS)
        train_norm_phy_losses = np.zeros(NUMEPOCHS)
        val_norm_phy_losses = np.zeros(NUMEPOCHS)
        test_norm_phy_losses = np.zeros(NUMEPOCHS)
        train_e_losses = np.zeros(NUMEPOCHS)
        val_e_losses = np.zeros(NUMEPOCHS)
        test_e_losses = np.zeros(NUMEPOCHS)
        train_all_losses = np.zeros(NUMEPOCHS)
        val_all_losses = np.zeros(NUMEPOCHS)
        test_all_losses = np.zeros(NUMEPOCHS)
        
        # write log file
        task_timestamp = self.str_now()
        statistics = \
        """
==========================================================================
Action: training model.
Time: %s
Task Id: %s

Number of Epochs:   %d
Train Batch Size:   %d
Test Batch Size:    %d
Optimizer:          %s

Training Loss:      %s
Test Loss:          %s

--------------------------------------------------------------------------
            
        """ % \
        (
            task_timestamp,
            self.current_hash,
            NUMEPOCHS,
            train_batch_size,
            test_batch_size,
            optimizer,
            train_param['train_loss'],
            train_param['test_loss'],
        )
        
        if verbose:
            print(statistics)
        
        # write log file
        log_file = open(
            self.path_log + 'log_' + self.current_hash + '.txt', 
            "a"
        )
        log_file.write(statistics)
        log_file.close()
        loss_file.write(
            'Time: %s\n' % task_timestamp
        )
        loss_file.write(
            'Task Id: %s\n' % self.current_hash
        )

        # training
        loss_file.write('Epoch \t Training \t Test \t\t Loss-Phy \t Loss-E \t Anealing Factor\n')
            
        e_coff = loss_param['lambda_e0']
        s_coff = loss_param['lambda_s']
        
        if loss_param['noise']:
            noise_param = loss_param['noise']
            mode = noise_param['mode']
            mean = noise_param['mean']
            var = noise_param['var']
            noise_decay = noise_param['decay']
        
        if loss_param['cyclical']:
            cyclic_param = loss_param['cyclical']
            cyclic_mode = cyclic_param['mode']
            cyclic_mean = cyclic_param['mean']
            amp = cyclic_param['amp']
            period = cyclic_param['period']
            cyclic_decay = cyclic_param['decay']
            if 'phase' in cyclic_param:
                phase = cyclic_param['phase']
            else:
                phase = 0.0
                
        # the progress bar
        if self.master_bar is not None:
            child_bar = progress_bar(range(NUMEPOCHS), parent=self.master_bar)
            self.master_bar.names = ['train', 'val', 'test']
        else:
            child_bar = range(NUMEPOCHS)

        # record when training started
        train_time = 0
        
        if 'break_loop_early' in train_param:
            if train_param['break_loop_early'] == False:
                stopped_epoch = NUMEPOCHS - 1
                
        for epoch in child_bar:
            model.train()
            start_time = time.time()    # start recording time
            if train_param['train_loss'] != []:
                for batchX, batchY, batchH in data_train_loader:  
                    # Forward pass
                    outputs = model(batchX)
                    
                    s_coff = loss_param['lambda_s']
                    
                    # cyclical pattern
                    if loss_param['cold_start']:
                        s_coff += self.cold_start(epoch=epoch, **loss_param['cold_start'])
                        s_coff = max(0, s_coff)
                    
                    # add noise
                    if loss_param['noise']:
                        noise = self.generate_noise(
                            mode=mode, mean=mean, var=var
                            )
                        s_coff += noise
                        s_coff = max(0, s_coff)

                    # cyclical pattern
                    if loss_param['cyclical']:
                        cyclic = self.cyclical(
                            mode=cyclic_mode, epoch=epoch, mean=cyclic_mean, amp=amp, period=period, phase=phase
                            )
                        s_coff += cyclic
                        s_coff = max(0, s_coff)
                    if 'vanilla' in train_param:
                        vanilla = train_param['vanilla']
                    else:
                        vanilla = False
                    loss = self.loss_func(
                        data, 
                        train_param['train_loss'],
                        outputs=outputs,
                        e_coff=e_coff,
                        s_coff=s_coff,
                        batchX=batchX, 
                        batchY=batchY,
                        batchH=batchH,
                        norm=loss_param['norm_wf'],
                        vanilla=vanilla
                    )[0]

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    if train_param['cyclical']:
                        scheduler.step()
                    else:
                        optimizer.step()
                    
            if train_param['test_loss'] != []:
                for batchX, batchY, batchH in data_test_loader:
                    # Forward pass
                    outputs = model(batchX)

                    s_coff = loss_param['lambda_s']
                    
                    # cyclical pattern
                    if loss_param['cold_start']:
                        s_coff += self.cold_start(epoch=epoch, **loss_param['cold_start'])
                        s_coff = max(0, s_coff)
                    
                    # add noise
                    if loss_param['noise']:
                        noise = self.generate_noise(
                            mode=mode, mean=mean, var=var
                            )
                        s_coff += noise
                        s_coff = max(0, s_coff)

                    # cyclical pattern
                    if loss_param['cyclical']:
                        cyclic = self.cyclical(
                            mode=cyclic_mode, epoch=epoch, mean=cyclic_mean, amp=amp, period=period, phase=phase
                            )
                        s_coff += cyclic
                        s_coff = max(0, s_coff)

                    loss = self.loss_func(
                        data, 
                        train_param['test_loss'],
                        outputs=outputs,
                        e_coff=e_coff,
                        s_coff=s_coff,
                        batchX=batchX, 
                        batchY=None,
                        batchH=batchH,
                        norm=loss_param['norm_wf'],
                    )[0]

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    if train_param['cyclical']:
                        scheduler.step()
                    else:
                        optimizer.step()
            
            end_time = time.time()     # end recording time
            train_time += end_time - start_time   # accumulate training time
            
            # record the loss history
            model.eval()
            train_losses[epoch] = criterion(
                model(data.X_train_tensor), data.y_train_tensor
            ).item()
            val_losses[epoch] = criterion(
                model(data.X_val_tensor), data.y_val_tensor
            ).item()
            test_losses[epoch] = criterion(
                model(data.X_test_tensor), data.y_test_tensor
            ).item()
            
            
            # train losses
            (loss, 
            train_phy_losses[epoch], 
            train_norm_phy_losses[epoch], 
            train_e_losses[epoch]) = self.loss_func(
                data, 
                train_param['train_loss'],
                outputs=model(data.X_train_tensor),
                e_coff=e_coff,
                s_coff=s_coff,
                batchX=data.X_train_tensor, 
                batchY=data.y_train_tensor,
                batchH=data.X_train_origin,
                norm=loss_param['norm_wf']
            )
            train_all_losses[epoch] = float(loss)
            
            # val losses
            (loss, 
            val_phy_losses[epoch], 
            val_norm_phy_losses[epoch], 
            val_e_losses[epoch]) = self.loss_func(
                data, 
                [], 
                outputs=model(data.X_val_tensor),
                e_coff=e_coff,
                s_coff=s_coff,
                batchX=data.X_val_tensor, 
                batchY=data.y_val_tensor,
                batchH=data.X_val_origin,
                norm=loss_param['norm_wf']
            )
            val_all_losses[epoch] = float(loss)
            
            # test losses
            (loss, 
            test_phy_losses[epoch], 
            test_norm_phy_losses[epoch], 
            test_e_losses[epoch]) = self.loss_func(
                data, 
                train_param['test_loss'],
                outputs=model(data.X_test_tensor),
                e_coff=e_coff,
                s_coff=s_coff,
                batchX=data.X_test_tensor, 
                batchY=data.y_test_tensor,
                batchH=data.X_test_origin,
                norm=loss_param['norm_wf']
            )
            test_all_losses = float(loss)
            
            if epoch % loss_param['anneal_interval'] == 0:
                e_coff *= loss_param['anneal_factor']
                if loss_param['noise']:
                    var *=  noise_decay
                if loss_param['cyclical']:
                    amp *= cyclic_decay

            if epoch % train_param['print_interval'] == 0:
                loss_file.write(
                    '%d \t %.8f \t %.8f \t %.8f \t %.8f \t %.8f\n' %
                    (
                        epoch, 
                        train_losses[epoch], 
                        test_losses[epoch], 
                        val_phy_losses[epoch], 
                        val_e_losses[epoch], 
                        e_coff
                    )
                )
            
            # plot loss curve
            if epoch % 1 == 0 and self.master_bar is not None and self.plot_flag:
                y_upper_bound = max(train_losses.max(), val_losses[0].max(), test_losses[0].max())
                x_axis = np.arange(epoch+1) + 1
                graphs = [[x_axis, train_losses[:epoch+1]], [x_axis, val_losses[:epoch+1]], [x_axis, test_losses[:epoch+1]]]
                x_bounds = [0, NUMEPOCHS]
                y_bounds = [0.0, y_upper_bound]
                self.master_bar.update_graph(graphs, x_bounds, y_bounds)
            
            # early stopping
            if train_param['early_stopping']:
                early_stopping(val_losses[epoch], model)
                if early_stopping.early_stop:
                    if 'break_loop_early' in train_param:
                        if train_param['break_loop_early'] == True:
                            break
                        else:
                            stopped_epoch = min(epoch, stopped_epoch)
                    else:
                        break
        
        # record when training stopped and calculate time           
        time_per_epoch = train_time / epoch
        
        if 'break_loop_early' in train_param:
            if train_param['break_loop_early'] == False:
                epoch = stopped_epoch
        
        # print loss in log files
        if verbose and self.master_bar is not None:
            self.master_bar.write('Training stopped at %d/%d.' %(epoch, NUMEPOCHS))
        loss_file.write('Training stopped at %d/%d.' %(epoch, NUMEPOCHS))
        loss_file.write('Training time: %f seconds.' % train_time)
        loss_file.write('\nTraining Complete')
        loss_file.write(
            '\n--------------------------------------------------------------------------\n'
        )
        loss_file.close()
        
        # data frame for losses
        df_loss = pd.DataFrame({
            'train_mse': train_losses,
            'val_mse': train_losses,
            'test_mse': test_losses,
            'train_phy': train_phy_losses,
            'val_phy': val_phy_losses,
            'test_phy': test_phy_losses,
            'train_norm_phy': train_norm_phy_losses,
            'train_e': train_e_losses,
            'val_e': val_e_losses,
            'test_e': test_e_losses,
            'val_norm_phy': val_norm_phy_losses,
            'test_norm_phy': test_norm_phy_losses,
            'train_all': train_all_losses,
            'val_all': val_all_losses,
            'test_all': test_all_losses
        })
        df_loss.to_csv(self.path_out + "losses_" + self.current_hash + ".csv", index=False)
        
        # training statistics to return
        train_stats = {
            'epoch': epoch,
            'train_time': train_time,
            'time_per_epoch': time_per_epoch
        }
        
        # save or load model
        if train_param['early_stopping']:
            model.load_state_dict(torch.load(path_to_model))
        else:
            torch.save(model.state_dict(), path_to_model)
        return model, train_stats
        
    def str_now(self, date=True, time=True):
        """ Return current time in string. """
        if date and time:
            return datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        else:
            if date:
                return datetime.now().strftime("%m-%d-%Y")
            if time:
                return datetime.now().strftime("%H:%M:%S")
    
    def loss_func(self, data, loss_list, outputs, e_coff=0.0, s_coff=1.0, batchX=None, batchY=None, batchH=None, norm=False, vanilla=False):
        """ 
        Set batchY to None when train on test set. 
        Set batchX to None when only use MSE.
        """
        # MSE Loss
        criterion = torch.nn.MSELoss()
        if (batchY is not None) and ('mse_loss' in loss_list):
            loss = criterion(outputs, batchY)
        else:
            loss = 0.0
            
        if batchH is not None:
            origin_input = batchH
        else:
            # inverse transformation for normalization
            if data.std_scaler_x is not None:
                origin_input = inverse_norm(
                    batchX, data.X_scale_tensor, data.X_mean_tensor
                )
            else:
                origin_input = batchX

        if data.std_scaler_y is not None:
            origin_output = inverse_norm(
                outputs, data.y_scale_tensor, data.y_mean_tensor
            )
            origin_y = inverse_norm(
                batchY, data.y_scale_tensor, data.y_mean_tensor
            )
        else:
            origin_output = outputs
            origin_y = batchY

        # physics loss and energy loss
        if 'phy_loss' in loss_list:
            loss_phy = phy_loss(
                origin_output,
                origin_y,
                origin_input,
                norm=norm
            )
        else:
            loss_phy = 0.0
            
        if 'energy_loss' in loss_list:
            loss_e = energy_loss(
                origin_output,
                origin_input
            )
        else:
            loss_e = 0.0
            
        if type(loss_phy) == torch.Tensor or type(loss_e) == torch.Tensor:
            if vanilla:
                losses = []
                if loss:
                    losses.append(loss)
                if torch.sum(loss_phy):
                    losses.append(torch.mean(s_coff * loss_phy))
                if torch.sum(loss_e):
                    losses.append(torch.mean(e_coff * loss_e))
                if not losses:
                    loss = 0.0
                else:
                    loss = np.random.choice(losses)
            else:
                loss += torch.mean(s_coff * loss_phy + e_coff * loss_e)
        
        norm_loss_phy = phy_loss(
            origin_output,
            origin_y,
            origin_input,
            norm=True
        )
        norm_loss_phy = torch.mean(norm_loss_phy).item()
        loss_phy = phy_loss(
            origin_output,
            origin_y,
            origin_input,
            norm=False
        )
        loss_phy = torch.mean(loss_phy).item()
        loss_e = energy_loss(
            origin_output,
            origin_input
        )
        loss_e = torch.mean(loss_e).item()
        return loss, loss_phy, norm_loss_phy, loss_e
    
    def evaluate(self, model, data, stats, param, name='unnamed'):
        """ Evaluate model and save data. """
        
        # eval model
        criterion = torch.nn.MSELoss()
        model.eval()
        
        pred_test = model(data.X_test_tensor).detach()
        true_test = data.y_test_tensor
        pred_val = model(data.X_val_tensor).detach()
        true_val = data.y_val_tensor
        pred_train = model(data.X_train_tensor).detach()
        true_train = data.y_train_tensor
        
        mse_all = criterion(pred_test, true_test)
        mse_wf = criterion(pred_test[:, :-1], true_test[:, :-1])
        mse_e = criterion(pred_test[:, -1], true_test[:, -1])
        
        mse_all_val = criterion(pred_val, true_val)
        mse_wf_val = criterion(pred_val[:, :-1], true_val[:, :-1])
        mse_e_val = criterion(pred_val[:, -1], true_val[:, -1])
        
        mse_all_train = criterion(pred_train, true_train)
        mse_wf_train = criterion(pred_train[:, :-1], true_train[:, :-1])
        mse_e_train = criterion(pred_train[:, -1], true_train[:, -1])
        
        overlap = torch.mean(torch.sum(pred_test * true_test, dim=1) / (
            torch.sqrt(torch.sum(true_test**2, dim=1)) * 
            torch.sqrt(torch.sum(pred_test**2, dim=1))
        ))
        
        overlap_val = torch.mean(torch.sum(pred_val * true_val, dim=1) / (
            torch.sqrt(torch.sum(true_val**2, dim=1)) * 
            torch.sqrt(torch.sum(pred_val**2, dim=1))
        ))
        
        overlap_train = torch.mean(torch.sum(pred_train * true_train, dim=1) / (
            torch.sqrt(torch.sum(true_train**2, dim=1)) * 
            torch.sqrt(torch.sum(pred_train**2, dim=1))
        ))
        
        # write loss file
        model_hash = hex(hash(model))
        path_to_model = '../models/%s.pt' % self.current_hash
        
        # dict to write in result file
        s = {
            'name': name,
            'model_hash': model_hash,
            'model_path': path_to_model,
            'log_path': self.path_log + 'log_' + self.current_hash + '.txt',
            'loss_path': self.path_log + 'loss_' + self.current_hash + '.txt',
            'loss_csv': self.path_out + 'losses_'  + self.current_hash + ".csv",
            'mse_all': mse_all.item(),
            'mse_wave_funtion': mse_wf.item(),
            'mse_energy': mse_e.item(),
            'wave_function_overlap': overlap.item(),
            'val_mse_all': mse_all_val.item(),
            'val_mse_wave_funtion': mse_wf_val.item(),
            'val_mse_energy': mse_e_val.item(),
            'val_wave_function_overlap': overlap_train.item(),
            'train_mse_all': mse_all_train.item(),
            'train_mse_wave_funtion': mse_wf_train.item(),
            'train_mse_energy': mse_e_train.item(),
            'train_wave_function_overlap': overlap_train.item(),
            'param_hash': hex(hash(str(param.__dict__))),
            'params': param.__dict__
        }
        
        # update result with training statistics
        s.update(stats)
        
        # write results
        result_file = open(
            self.path_out + 'result_%s_%s.txt' % (self.str_now(), self.current_hash), 
            "a"
        )
        result_file.write(str(s))
        result_file.close()
        
    def generate_noise(self, mode='guassian', mean=0.0, var=1.0):
        if mode == 'guassian':
            return np.random.normal(mean, var)
        if mode == 'uniform':
            return np.random.uniform(mean-np.sqrt(var), mean+np.sqrt(var))

    def cyclical(self, epoch, mode='sin', mean=1.0, amp=1.0, period=20, phase=0.0):
        if mode == 'sin':
            return amp / 2 * np.sin(2 * np.pi * epoch / period + phase) + mean
        elif mode == 'cos':
            return amp / 2 * np.cos(2 * np.pi * epoch / period + phase) + mean
        elif mode == 'triangular':
            return amp / 2 * signal.sawtooth(2 * np.pi * epoch / period + phase) + mean
    
    def cold_start(self, epoch, mode='sigmoid', lambda_s=0.5, threshold=50, smooth=0.1):
        if mode == 'sigmoid':
            sigmoid = lambda x : 1/(1 + np.exp(-x))
            return np.maximum(lambda_s * (sigmoid((epoch - threshold) * smooth)), 0)
        elif mode == 'inverse-sigmoid':
            sigmoid = lambda x : 1/(1 + np.exp(-x))
            return np.maximum(lambda_s * (1 - (sigmoid((epoch - threshold) * smooth))), 0)
        elif mode == 'quick-start':
            s_coeff = lambda s, alpha, threshold, epoch: s * (1 - (1 + alpha) ** np.minimum(- epoch + threshold, 0))
            return np.maximum(s_coeff(lambda_s, smooth, threshold, epoch), 0)
        elif mode == 'quick-drop':
            s_coeff = lambda s, alpha, threshold, epoch: s * ((1 + alpha) ** np.minimum(- epoch + threshold, 0))
            return np.maximum(s_coeff(lambda_s, smooth, threshold, epoch), 0)
        