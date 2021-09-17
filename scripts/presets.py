from parameters import Params

import re
import glob
import os
import json
import numpy as np
import sys
sys.path.append('//home/jayroxis/Condensed Matter Theory/scripts/')

class LambdaSearch(object):
    """
      ===============================================
        Registered Models:
      -----------------------------------------------
        Deep Neural Networks        |   DNN
        Shrodinger Loss DNN         |   S-DNN
        Energy Loss S-DNN           |   SE-DNN
        Normalized SE-DNN           |   NSE-DNN
        Extended Training S-DNN     |   S-DNNex
        Extended Training SE-DNN    |   SE-DNNex
        Normalized S-DNNex          |   NS-DNNex
        Normalized SE-DNNex         |   NSE-DNNex
        Label-Free S-DNNex          |   S-DNNex-LB
        Label-Free SE-DNNex         |   SE-DNNex-LB
        Normalized S-DNNex-LB       |   NS-DNNex-LB
        Normalized SE-DNNex-LB      |   NSE-DNNex-LB
      ===============================================
    
    """

    def __init__(self, data_path='//home/elhamod/melhamodenv/CMT/downloadedData/Ising/'):
        self.data_path = data_path
    
    def DNN(self, patience=50, anneal=(0.5, 1.0), lambda_e0=(0.0, 5.0), lambda_s=(0.0, 10.0)):
        
        """ The baseline blackbox DNN. """
        
        param = Params()
        param.data_params['data_path'] = self.data_path

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        param.train_params['early_stopping'] = {'patience':patience, 'verbose':False, 'delta':0}
        param.train_params['cyclical'] = {}    # cyclical learning rate

        # coefficient for energy loss and shrodinger loss
        param.loss_params['anneal_interval'] = 10
        param.loss_params['anneal_factor'] = np.random.uniform(low=anneal[0], high=anneal[1])
        param.loss_params['lambda_e0'] = np.random.uniform(low=lambda_e0[0], high=lambda_e0[1])
        param.loss_params['lambda_s'] = np.random.uniform(low=lambda_s[0], high=lambda_s[1])
        
        # for coefficient of the shrodinger loss
        param.loss_params['noise'] = {}       # noisy coefficient
        param.loss_params['cyclical'] = {}    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        param.loss_params['norm_wf'] = False
        param.train_params['train_loss'] = [
            'mse_loss',
        ]
        param.train_params['test_loss'] = []
        return param
    
    def DNN_overlap(self, patience=50, anneal=(0.5, 1.0), lambda_e0=(0.0, 5.0), lambda_s=(0.0, 10.0)):
        
        """ The baseline blackbox DNN. """
        
        param = Params()
        param.data_params['data_path'] = self.data_path

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        param.train_params['early_stopping'] = {'patience':patience, 'verbose':False, 'delta':0}
        param.train_params['cyclical'] = {}    # cyclical learning rate

        # coefficient for energy loss and shrodinger loss
        param.loss_params['anneal_interval'] = 10
        param.loss_params['anneal_factor'] = np.random.uniform(low=anneal[0], high=anneal[1])
        param.loss_params['lambda_e0'] = np.random.uniform(low=lambda_e0[0], high=lambda_e0[1])
        param.loss_params['lambda_s'] = np.random.uniform(low=lambda_s[0], high=lambda_s[1])
        
        # for coefficient of the shrodinger loss
        param.loss_params['noise'] = {}       # noisy coefficient
        param.loss_params['cyclical'] = {}    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        param.loss_params['norm_wf'] = False
        param.train_params['train_loss'] = [
            'overlap_loss',
        ]
        param.train_params['test_loss'] = []
        return param
    
    
    def S_DNN(self, patience=50, anneal=(0.5, 1.0), lambda_e0=(0.0, 5.0), lambda_s=(0.0, 10.0)):
        
        """ Only Shrodinger loss and only on training set. """
        
        param = Params()
        param.data_params['data_path'] = self.data_path

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        param.train_params['early_stopping'] = {'patience':patience, 'verbose':False, 'delta':0}
        param.train_params['cyclical'] = {}    # cyclical learning rate

        # coefficient for energy loss and shrodinger loss
        param.loss_params['anneal_interval'] = 10
        param.loss_params['anneal_factor'] = np.random.uniform(low=anneal[0], high=anneal[1])
        param.loss_params['lambda_e0'] = np.random.uniform(low=lambda_e0[0], high=lambda_e0[1])
        param.loss_params['lambda_s'] = np.random.uniform(low=lambda_s[0], high=lambda_s[1])
        
        # for coefficient of the shrodinger loss
        param.loss_params['noise'] = {}       # noisy coefficient
        param.loss_params['cyclical'] = {}    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        param.loss_params['norm_wf'] = False
        param.train_params['train_loss'] = [
            'mse_loss',
            'phy_loss', 
        ]
        param.train_params['test_loss'] = []
        return param
    
    def SE_DNN(self, patience=50, anneal=(0.5, 1.0), lambda_e0=(0.0, 5.0), lambda_s=(0.0, 10.0)):
        
        """ S, E loss only on training set. """
        
        param = Params()
        param.data_params['data_path'] = self.data_path

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        param.train_params['early_stopping'] = {'patience':patience, 'verbose':False, 'delta':0}
        param.train_params['cyclical'] = {}    # cyclical learning rate

        # coefficient for energy loss and shrodinger loss
        param.loss_params['anneal_interval'] = 10
        param.loss_params['anneal_factor'] = np.random.uniform(low=anneal[0], high=anneal[1])
        param.loss_params['lambda_e0'] = np.random.uniform(low=lambda_e0[0], high=lambda_e0[1])
        param.loss_params['lambda_s'] = np.random.uniform(low=lambda_s[0], high=lambda_s[1])
        
        # for coefficient of the shrodinger loss
        param.loss_params['noise'] = {}       # noisy coefficient
        param.loss_params['cyclical'] = {}    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        param.loss_params['norm_wf'] = False
        param.train_params['train_loss'] = [
            'mse_loss',
            'phy_loss',
            'energy_loss'
        ]
        param.train_params['test_loss'] = []
        return param

    def NSE_DNN(self, patience=50, anneal=(0.5, 1.0), lambda_e0=(0.0, 5.0), lambda_s=(0.0, 10.0)):
        
        """ S, E loss only on training set. """
        
        param = Params()
        param.data_params['data_path'] = self.data_path

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        param.train_params['early_stopping'] = {'patience':patience, 'verbose':False, 'delta':0}
        param.train_params['cyclical'] = {}    # cyclical learning rate

        # coefficient for energy loss and shrodinger loss
        param.loss_params['anneal_interval'] = 10
        param.loss_params['anneal_factor'] = np.random.uniform(low=anneal[0], high=anneal[1])
        param.loss_params['lambda_e0'] = np.random.uniform(low=lambda_e0[0], high=lambda_e0[1])
        param.loss_params['lambda_s'] = np.random.uniform(low=lambda_s[0], high=lambda_s[1])
        
        # for coefficient of the shrodinger loss
        param.loss_params['noise'] = {}       # noisy coefficient
        param.loss_params['cyclical'] = {}    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        param.loss_params['norm_wf'] = True
        param.train_params['train_loss'] = [
            'mse_loss',
            'phy_loss',
            'energy_loss'
        ]
        param.train_params['test_loss'] = []
        return param
    
    def S_DNNex(self, patience=50, anneal=(0.5, 1.0), lambda_e0=(0.0, 5.0), lambda_s=(0.0, 10.0)):
        
        """ Only Shrodinger loss. """
        
        param = Params()
        param.data_params['data_path'] = self.data_path

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        param.train_params['early_stopping'] = {'patience':patience, 'verbose':False, 'delta':0}
        param.train_params['cyclical'] = {}    # cyclical learning rate

        # coefficient for energy loss and shrodinger loss
        param.loss_params['anneal_interval'] = 10
        param.loss_params['anneal_factor'] = np.random.uniform(low=anneal[0], high=anneal[1])
        param.loss_params['lambda_e0'] = np.random.uniform(low=lambda_e0[0], high=lambda_e0[1])
        param.loss_params['lambda_s'] = np.random.uniform(low=lambda_s[0], high=lambda_s[1])
        
        # for coefficient of the shrodinger loss
        param.loss_params['noise'] = {}       # noisy coefficient
        param.loss_params['cyclical'] = {}    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        param.loss_params['norm_wf'] = False
        param.train_params['train_loss'] = [
            'mse_loss',
            'phy_loss', 
        ]
        param.train_params['test_loss'] = [
            'phy_loss', 
        ]
        return param
    
    def SE_DNNex(self, patience=50, anneal=(0.5, 1.0), lambda_e0=(0.0, 5.0), lambda_s=(0.0, 10.0)):
        
        """ The best performed model. """
        
        param = Params()
        param.data_params['data_path'] = self.data_path

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        param.train_params['early_stopping'] = {'patience':patience, 'verbose':False, 'delta':0}
        param.train_params['cyclical'] = {}    # cyclical learning rate

        # coefficient for energy loss and shrodinger loss
        param.loss_params['anneal_interval'] = 10
        param.loss_params['anneal_factor'] = np.random.uniform(low=anneal[0], high=anneal[1])
        param.loss_params['lambda_e0'] = np.random.uniform(low=lambda_e0[0], high=lambda_e0[1])
        param.loss_params['lambda_s'] = np.random.uniform(low=lambda_s[0], high=lambda_s[1])
        
        # for coefficient of the shrodinger loss
        param.loss_params['noise'] = {}       # noisy coefficient
        param.loss_params['cyclical'] = {}    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        param.loss_params['norm_wf'] = False
        param.train_params['train_loss'] = [
            'mse_loss',
            'phy_loss', 
            'energy_loss'
        ]
        param.train_params['test_loss'] = [
            'phy_loss', 
            'energy_loss'
        ]
        return param
    
    def NS_DNNex(self, patience=50, anneal=(0.5, 1.0), lambda_e0=(0.0, 5.0), lambda_s=(0.0, 10.0)):
        
        """ See what will happen without energy loss. """
        
        param = Params()
        param.data_params['data_path'] = self.data_path

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        param.train_params['early_stopping'] = {'patience':patience, 'verbose':False, 'delta':0}
        param.train_params['cyclical'] = {}    # cyclical learning rate

        # coefficient for energy loss and shrodinger loss
        param.loss_params['anneal_interval'] = 10
        param.loss_params['anneal_factor'] = np.random.uniform(low=anneal[0], high=anneal[1])
        param.loss_params['lambda_e0'] = np.random.uniform(low=lambda_e0[0], high=lambda_e0[1])
        param.loss_params['lambda_s'] = np.random.uniform(low=lambda_s[0], high=lambda_s[1])
        
        # for coefficient of the shrodinger loss
        param.loss_params['noise'] = {}       # noisy coefficient
        param.loss_params['cyclical'] = {}    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        param.loss_params['norm_wf'] = True
        param.train_params['train_loss'] = [
            'mse_loss',
            'phy_loss', 
        ]
        param.train_params['test_loss'] = [
            'phy_loss', 
        ]
        return param
    
    def NSE_DNNex(self, patience=50, anneal=(0.5, 1.0), lambda_e0=(0.0, 5.0), lambda_s=(0.0, 10.0)):
        
        """ Normalized version. """
        
        param = Params()
        param.data_params['data_path'] = self.data_path

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        param.train_params['early_stopping'] = {'patience':patience, 'verbose':False, 'delta':0}
        param.train_params['cyclical'] = {}    # cyclical learning rate

        # coefficient for energy loss and shrodinger loss
        param.loss_params['anneal_interval'] = 10
        param.loss_params['anneal_factor'] = np.random.uniform(low=anneal[0], high=anneal[1])
        param.loss_params['lambda_e0'] = np.random.uniform(low=lambda_e0[0], high=lambda_e0[1])
        param.loss_params['lambda_s'] = np.random.uniform(low=lambda_s[0], high=lambda_s[1])
        
        # for coefficient of the shrodinger loss
        param.loss_params['noise'] = {}       # noisy coefficient
        param.loss_params['cyclical'] = {}    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        param.loss_params['norm_wf'] = True
        param.train_params['train_loss'] = [
            'mse_loss',
            'phy_loss', 
            'energy_loss'
        ]
        param.train_params['test_loss'] = [
            'phy_loss', 
            'energy_loss'
        ]
        return param
    
    def NSE_DNNex_overlap(self, patience=50, anneal=(0.5, 1.0), lambda_e0=(0.0, 5.0), lambda_s=(0.0, 10.0), mse=False):
        
        """ Normalized version. """
        param = Params()
        param.data_params['data_path'] = self.data_path

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        param.train_params['early_stopping'] = {'patience':patience, 'verbose':False, 'delta':0}
        param.train_params['cyclical'] = {}    # cyclical learning rate

        # coefficient for energy loss and shrodinger loss
        param.loss_params['anneal_interval'] = 10
        param.loss_params['anneal_factor'] = np.random.uniform(low=anneal[0], high=anneal[1])
        param.loss_params['lambda_e0'] = np.random.uniform(low=lambda_e0[0], high=lambda_e0[1])
        param.loss_params['lambda_s'] = np.random.uniform(low=lambda_s[0], high=lambda_s[1])
        
        # for coefficient of the shrodinger loss
        param.loss_params['noise'] = {}       # noisy coefficient
        param.loss_params['cyclical'] = {}    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        param.loss_params['norm_wf'] = True
        param.train_params['train_loss'] = [
            'overlap_loss',
            'phy_loss', 
            'energy_loss'
        ]
        if mse == True: param.train_params['train_loss'] = param.train_params['train_loss'] + ['mse_loss']
        param.train_params['test_loss'] = [
            'phy_loss', 
            'energy_loss'
        ]
        return param
    
    def S_DNNex_LB(self, patience=50, anneal=(0.5, 1.0), lambda_e0=(0.0, 5.0), lambda_s=(0.0, 10.0)):
        
        """ The label-free S-DNNex. """
        
        param = Params()
        param.data_params['data_path'] = self.data_path

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        param.train_params['early_stopping'] = {'patience':patience, 'verbose':False, 'delta':0}
        param.train_params['cyclical'] = {}    # cyclical learning rate

        # coefficient for energy and shrodinger loss
        param.loss_params['anneal_interval'] = 10
        param.loss_params['anneal_factor'] = np.random.uniform(low=anneal[0], high=anneal[1])
        param.loss_params['lambda_e0'] = np.random.uniform(low=lambda_e0[0], high=lambda_e0[1])
        param.loss_params['lambda_s'] = np.random.uniform(low=lambda_s[0], high=lambda_s[1])

        # for coefficient of the shrodinger loss
        param.loss_params['noise'] = {}       # noisy coefficient
        param.loss_params['cyclical'] = {}    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        param.loss_params['norm_wf'] = False
        param.train_params['train_loss'] = [
            'phy_loss', 
        ]
        param.train_params['test_loss'] = [
            'phy_loss', 
        ]
        return param
    
    def SE_DNNex_LB(self, patience=50, anneal=(0.5, 1.0), lambda_e0=(0.0, 5.0), lambda_s=(0.0, 10.0)):
        
        """ The label-free model. """
        
        param = Params()
        param.data_params['data_path'] = self.data_path

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        param.train_params['early_stopping'] = {'patience':patience, 'verbose':False, 'delta':0}
        param.train_params['cyclical'] = {}    # cyclical learning rate

        # coefficient for energy and shrodinger loss
        param.loss_params['anneal_interval'] = 10
        param.loss_params['anneal_factor'] = np.random.uniform(low=anneal[0], high=anneal[1])
        param.loss_params['lambda_e0'] = np.random.uniform(low=lambda_e0[0], high=lambda_e0[1])
        param.loss_params['lambda_s'] = np.random.uniform(low=lambda_s[0], high=lambda_s[1])

        # for coefficient of the shrodinger loss
        param.loss_params['noise'] = {}       # noisy coefficient
        param.loss_params['cyclical'] = {}    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        param.loss_params['norm_wf'] = False
        param.train_params['train_loss'] = [
            'phy_loss',
            'energy_loss'
        ]
        param.train_params['test_loss'] = [
            'phy_loss',
            'energy_loss'
        ]
        return param
        
    def NS_DNNex_LB(self, patience=50, anneal=(0.5, 1.0), lambda_e0=(0.0, 5.0), lambda_s=(0.0, 10.0)):
        
        """ The label-free model. """
        
        param = Params()
        param.data_params['data_path'] = self.data_path

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        param.train_params['early_stopping'] = {'patience':patience, 'verbose':False, 'delta':0}
        param.train_params['cyclical'] = {}    # cyclical learning rate

        # coefficient for energy and shrodinger loss
        param.loss_params['anneal_interval'] = 10
        param.loss_params['anneal_factor'] = np.random.uniform(low=anneal[0], high=anneal[1])
        param.loss_params['lambda_e0'] = np.random.uniform(low=lambda_e0[0], high=lambda_e0[1])
        param.loss_params['lambda_s'] = np.random.uniform(low=lambda_s[0], high=lambda_s[1])

        # for coefficient of the shrodinger loss
        param.loss_params['noise'] = {}       # noisy coefficient
        param.loss_params['cyclical'] = {}    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        param.loss_params['norm_wf'] = True
        param.train_params['train_loss'] = [
            'phy_loss', 
        ]
        param.train_params['test_loss'] = [
            'phy_loss', 
        ]
        return param

    def NSE_DNNex_LB(self, patience=50, anneal=(0.5, 1.0), lambda_e0=(0.0, 5.0), lambda_s=(0.0, 10.0)):
        
        """ The label-free model. """
        
        param = Params()
        param.data_params['data_path'] = self.data_path

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        param.train_params['early_stopping'] = {'patience':patience, 'verbose':False, 'delta':0}
        param.train_params['cyclical'] = {}    # cyclical learning rate

        # coefficient for energy and shrodinger loss
        param.loss_params['anneal_interval'] = 10
        param.loss_params['anneal_factor'] = np.random.uniform(low=anneal[0], high=anneal[1])
        param.loss_params['lambda_e0'] = np.random.uniform(low=lambda_e0[0], high=lambda_e0[1])
        param.loss_params['lambda_s'] = np.random.uniform(low=lambda_s[0], high=lambda_s[1])

        # for coefficient of the shrodinger loss
        param.loss_params['noise'] = {}       # noisy coefficient
        param.loss_params['cyclical'] = {}    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        param.loss_params['norm_wf'] = True
        param.train_params['train_loss'] = [
            'phy_loss', 
            'energy_loss'
        ]
        param.train_params['test_loss'] = [
            'phy_loss', 
            'energy_loss'
        ]
        return param
    
    def vNSE_DNNex(self, patience=50, anneal=(0.5, 1.0), lambda_e0=(0.0, 5.0), lambda_s=(0.0, 10.0)):
        
        """ Normalized version. """
        
        param = Params()
        param.data_params['data_path'] = self.data_path

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        param.train_params['early_stopping'] = {'patience':patience, 'verbose':False, 'delta':0}
        param.train_params['cyclical'] = {}    # cyclical learning rate

        # coefficient for energy loss and shrodinger loss
        param.loss_params['anneal_interval'] = 10
        param.loss_params['anneal_factor'] = np.random.uniform(low=anneal[0], high=anneal[1])
        param.loss_params['lambda_e0'] = np.random.uniform(low=lambda_e0[0], high=lambda_e0[1])
        param.loss_params['lambda_s'] = np.random.uniform(low=lambda_s[0], high=lambda_s[1])
        
        # for coefficient of the shrodinger loss
        param.loss_params['noise'] = {}       # noisy coefficient
        param.loss_params['cyclical'] = {}    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        param.loss_params['norm_wf'] = True
        param.train_params['train_loss'] = [
            'mse_loss',
            'phy_loss', 
            'energy_loss'
        ]
        param.train_params['test_loss'] = [
            'phy_loss', 
            'energy_loss'
        ]
        param.train_params['vanilla'] = True
        return param

# this class is used for dict to object
class ParamObject(object):
    def __init__(self, d):
        self.__dict__ = d    
    
    
class LoadFilter(object):
    """
      ===============================================
        Registered Models:
      -----------------------------------------------
        Deep Neural Networks        |   DNN
        Shrodinger Loss DNN         |   S-DNN
        Energy Loss S-DNN           |   SE-DNN
        Normalized SE-DNN           |   NSE-DNN
        Extended Training S-DNN     |   S-DNNex
        Extended Training SE-DNN    |   SE-DNNex
        Normalized S-DNNex          |   NS-DNNex
        Normalized SE-DNNex         |   NSE-DNNex
        Label-Free S-DNNex          |   S-DNNex-LB
        Label-Free SE-DNNex         |   SE-DNNex-LB
        Normalized S-DNNex-LB       |   NS-DNNex-LB
        Normalized SE-DNNex-LB      |   NSE-DNNex-LB
        Cyclical Lambda-S NSE-DNNex |   C-NSE-DNNex
      ===============================================
    
    """
    def __init__(self):
        pass

    def DNN(self, d):
                
        """ The baseline blackbox DNN. """
        if 'params' in d:
            param = ParamObject(d['params'])

        flag = True

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        flag = flag and not param.train_params['cyclical']    # cyclical learning rate
            
        # for coefficient of the shrodinger loss
        flag = flag and not param.loss_params['noise']       # noisy coefficient
        flag = flag and not param.loss_params['cyclical']    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        flag = flag and not param.loss_params['norm_wf']
        flag = flag and param.train_params['train_loss'] == [
            'mse_loss',
        ]
        flag = flag and param.train_params['test_loss'] == []
        return flag
    
    def S_DNN(self, d):
        
        """ Only Shrodinger loss and only on training set. """
        if 'params' in d:
            param = ParamObject(d['params'])

        flag = True

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        flag = flag and not param.train_params['cyclical']    # cyclical learning rate
            
        # for coefficient of the shrodinger loss
        flag = flag and not param.loss_params['noise']       # noisy coefficient
        flag = flag and not param.loss_params['cyclical']    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        flag = flag and not param.loss_params['norm_wf']
        flag = flag and param.train_params['train_loss'] == [
            'mse_loss',
            'phy_loss', 
        ]
        flag = flag and param.train_params['test_loss'] == []
        return flag
    
    def SE_DNN(self, d):
        
        """ S, E loss only on training set. """
        if 'params' in d:
            param = ParamObject(d['params'])

        flag = True

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        flag = flag and not param.train_params['cyclical']    # cyclical learning rate
            
        # for coefficient of the shrodinger loss
        flag = flag and not param.loss_params['noise']       # noisy coefficient
        flag = flag and not param.loss_params['cyclical']    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        flag = flag and not param.loss_params['norm_wf']
        flag = flag and param.train_params['train_loss'] == [
            'mse_loss',
            'phy_loss',
            'energy_loss'
        ]
        flag = flag and param.train_params['test_loss'] == []
        return flag

    def NSE_DNN(self, d):
        
        """ S, E loss only on training set. """
        if 'params' in d:
            param = ParamObject(d['params'])

        flag = True

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        flag = flag and not param.train_params['cyclical']    # cyclical learning rate
            
        # for coefficient of the shrodinger loss
        flag = flag and not param.loss_params['noise']       # noisy coefficient
        flag = flag and not param.loss_params['cyclical']    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        flag = flag and param.loss_params['norm_wf']
        flag = flag and param.train_params['train_loss'] == [
            'mse_loss',
            'phy_loss',
            'energy_loss'
        ]
        flag = flag and param.train_params['test_loss'] == []
        return flag
    
    def S_DNNex(self, d):
        
        """ Only Shrodinger loss. """
        if 'params' in d:
            param = ParamObject(d['params'])

        flag = True

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        flag = flag and not param.train_params['cyclical']    # cyclical learning rate
            
        # for coefficient of the shrodinger loss
        flag = flag and not param.loss_params['noise']       # noisy coefficient
        flag = flag and not param.loss_params['cyclical']    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        flag = flag and not param.loss_params['norm_wf']
        flag = flag and param.train_params['train_loss'] == [
            'mse_loss',
            'phy_loss', 
        ]
        flag = flag and param.train_params['test_loss'] == [
            'phy_loss', 
        ]
        return flag
    
    def SE_DNNex(self, d):
        
        """ The best performed model. """
        if 'params' in d:
            param = ParamObject(d['params'])

        flag = True

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        flag = flag and not param.train_params['cyclical']    # cyclical learning rate
            
        # for coefficient of the shrodinger loss
        flag = flag and not param.loss_params['noise']       # noisy coefficient
        flag = flag and not param.loss_params['cyclical']    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        flag = flag and not param.loss_params['norm_wf']
        flag = flag and param.train_params['train_loss'] == [
            'mse_loss',
            'phy_loss', 
            'energy_loss'
        ]
        flag = flag and param.train_params['test_loss'] == [
            'phy_loss', 
            'energy_loss'
        ]
        return flag
    
    def NS_DNNex(self, d):
        
        """ See what will happen without energy loss. """
        if 'params' in d:
            param = ParamObject(d['params'])

        flag = True

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        flag = flag and not param.train_params['cyclical']    # cyclical learning rate
            
        # for coefficient of the shrodinger loss
        flag = flag and not param.loss_params['noise']       # noisy coefficient
        flag = flag and not param.loss_params['cyclical']    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        flag = flag and param.loss_params['norm_wf']
        flag = flag and param.train_params['train_loss'] == [
            'mse_loss',
            'phy_loss', 
        ]
        flag = flag and param.train_params['test_loss'] == [
            'phy_loss', 
        ]
        return flag
    
    def NSE_DNNex(self, d):
        
        """ Normalized version. """
        if 'params' in d:
            param = ParamObject(d['params'])

        flag = True

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        flag = flag and not param.train_params['cyclical']    # cyclical learning rate
            
        # for coefficient of the shrodinger loss
        flag = flag and not param.loss_params['noise']       # noisy coefficient
        flag = flag and not param.loss_params['cyclical']    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        flag = flag and param.loss_params['norm_wf']
        flag = flag and param.train_params['train_loss'] == [
            'mse_loss',
            'phy_loss', 
            'energy_loss'
        ]
        flag = flag and param.train_params['test_loss'] == [
            'phy_loss', 
            'energy_loss'
        ]
        return flag
    
    def S_DNNex_LB(self, d):
        
        """ The label-free S-DNNex. """
        if 'params' in d:
            param = ParamObject(d['params'])

        flag = True

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        flag = flag and not param.train_params['cyclical']    # cyclical learning rate

        # for coefficient of the shrodinger loss
        flag = flag and not param.loss_params['noise']       # noisy coefficient
        flag = flag and not param.loss_params['cyclical']    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        flag = flag and not param.loss_params['norm_wf']
        flag = flag and param.train_params['train_loss'] == [
            'phy_loss', 
        ]
        flag = flag and param.train_params['test_loss'] == [
            'phy_loss', 
        ]
        return flag
    
    def SE_DNNex_LB(self, d):
        
        """ The label-free model. """
        if 'params' in d:
            param = ParamObject(d['params'])

        flag = True

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        flag = flag and not param.train_params['cyclical']    # cyclical learning rate

        # for coefficient of the shrodinger loss
        flag = flag and not param.loss_params['noise']       # noisy coefficient
        flag = flag and not param.loss_params['cyclical']    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        flag = flag and not param.loss_params['norm_wf']
        flag = flag and param.train_params['train_loss'] == [
            'phy_loss',
            'energy_loss'
        ]
        flag = flag and param.train_params['test_loss'] == [
            'phy_loss',
            'energy_loss'
        ]
        return flag
        
    def NS_DNNex_LB(self, d):
        
        """ The label-free model. """
        if 'params' in d:
            param = ParamObject(d['params'])

        flag = True

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        flag = flag and not param.train_params['cyclical']    # cyclical learning rate

        # for coefficient of the shrodinger loss
        flag = flag and not param.loss_params['noise']       # noisy coefficient
        flag = flag and not param.loss_params['cyclical']    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        flag = flag and param.loss_params['norm_wf']
        flag = flag and param.train_params['train_loss'] == [
            'phy_loss', 
        ]
        flag = flag and param.train_params['test_loss'] == [
            'phy_loss', 
        ]
        return flag

    def NSE_DNNex_LB(self, d):
        
        """ The label-free model. """
        if 'params' in d:
            param = ParamObject(d['params'])

        flag = True

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        flag = flag and not param.train_params['cyclical']    # cyclical learning rate

        # for coefficient of the shrodinger loss
        flag = flag and not param.loss_params['noise']       # noisy coefficient
        flag = flag and not param.loss_params['cyclical']    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        flag = flag and param.loss_params['norm_wf']
        flag = flag and param.train_params['train_loss'] == [
            'phy_loss', 
            'energy_loss'
        ]
        flag = flag and param.train_params['test_loss'] == [
            'phy_loss', 
            'energy_loss'
        ]
        return flag

    def C_NSE_DNNex(self, d):
        
        """ Normalized version. """
        if 'params' in d:
            param = ParamObject(d['params'])

        flag = True

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        flag = flag and not param.train_params['cyclical']    # cyclical learning rate
            
        # for coefficient of the shrodinger loss
        flag = flag and not param.loss_params['noise']       # noisy coefficient
        flag = flag and param.loss_params['cyclical']    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        flag = flag and param.loss_params['norm_wf']
        flag = flag and param.train_params['train_loss'] == [
            'mse_loss',
            'phy_loss', 
            'energy_loss'
        ]
        flag = flag and param.train_params['test_loss'] == [
            'phy_loss', 
            'energy_loss'
        ]
        return flag
    
    def vNSE_DNNex(self, d):
        
        """ Vanilla stochastic Multitask. """
        if 'params' in d:
            param = ParamObject(d['params'])

        flag = True

        # set patience to 500 to give best potential to all models when doing hyper-parameter search
        flag = flag and not param.train_params['cyclical']    # cyclical learning rate
            
        # for coefficient of the shrodinger loss
        flag = flag and not param.loss_params['noise']       # noisy coefficient
        flag = flag and param.loss_params['cyclical']    # cyclical coefficient

        # loss function for NSE-DNNex-LB (label free)
        flag = flag and param.loss_params['norm_wf']
        flag = flag and param.train_params['train_loss'] == [
            'mse_loss',
            'phy_loss', 
            'energy_loss'
        ]
        flag = flag and param.train_params['test_loss'] == [
            'phy_loss', 
            'energy_loss'
        ]
        flag = flag and 'vanilla' in param.train_params
        flag = flag and param.train_params['vanilla']
        return flag