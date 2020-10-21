
import torch
import numpy as np
from torch.autograd import Variable

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import os
from scipy.io import loadmat
import h5py

def readMatFile(file_path, name):
    try:
        return loadmat(file_path)[name]
    except Exception as e:
        try:
            f = h5py.File(file_path, 'r')
            r = f.get(name)
            r = np.transpose(r)
            return r
        except Exception as e:
            print(e)


# train_test = "training", "validation", "testing"
def getData(path_to_data, train_test):
    directory = os.path.join(path_to_data,train_test)

    # read the files
    H = readMatFile(os.path.join(directory,'H.mat'), 'H')
    X = readMatFile(os.path.join(directory,'X.mat'), 'x')
    Y = readMatFile(os.path.join(directory,'Y.mat'), 'y')

    return H, X, Y
  


# ------------> data loader start here <----------------
class DatasetLoader(object):
    def __init__(
        self, 
        data_path, 
        train_size, 
    ):
        
        # store parameters
        data_path = os.path.join(data_path, train_size)
        self.data_path = data_path
        # load training data
        training_set = getData(
            path_to_data=data_path,
            train_test='training',
        )
        
        # load test data
        test_set = getData(
            path_to_data=data_path,
            train_test='testing',
        )
        val_set = getData(
            path_to_data=data_path,
            train_test='validation',
        )
        
        # interpret training variables
        Ham_val = val_set[0]
        input_val = val_set[1]
        targets_val = val_set[2][:,2:]
        eigenvals_val = val_set[2][:, 0:2]
        
        # phased-locked validation dataset: always positive psi_0
        self.val_X = input_val
        self.val_H = Ham_val
        self.val_y = np.concatenate((targets_val, eigenvals_val),axis=1)
        self.val_y = torch.tensor(self.val_y).float()
        self.available_val = Ham_val.shape[0]
        self.sign_val = torch.sign(self.val_y[:, 0]).view(-1, 1)
        self.val_y[:, :-2] *= self.sign_val
        
        # interpret training variables
        Ham_train = training_set[0]
        input_train = training_set[1]
        targets_train = training_set[2][:,2:]
        eigenvals_train = training_set[2][:, 0:2]
        
        
        # interpret test variables
        Ham_test = test_set[0]
        input_test = test_set[1]
        targets_test = test_set[2][:,2:]
        eigenvals_test = test_set[2][:, 0:2]

        # phased-locked training dataset: always positive psi_0
        self.train_X = input_train
        self.train_H = Ham_train
        self.train_y = np.concatenate((targets_train, eigenvals_train),axis=1)
        self.train_y = torch.tensor(self.train_y).float()
        self.available_train = Ham_train.shape[0]
        self.sign_train = torch.sign(self.train_y[:, 0]).view(-1, 1)
        self.train_y[:, :-2] *= self.sign_train
        
        # phased-locked test dataset: always positive psi_0
        self.test_X = input_test
        self.test_H = Ham_test
        self.test_y = np.concatenate((targets_test, eigenvals_test),axis=1)
        self.test_y = torch.tensor(self.test_y).float()
        self.available_test = Ham_test.shape[0]
        self.sign_test = torch.sign(self.test_y[:, 0]).view(-1, 1)
        self.test_y[:, :-2] *= self.sign_test
       
        # dimensions
        self.x_dim = self.train_X.shape[1]
        self.y_dim = self.train_y.shape[1]
        self.h_height = self.train_H.shape[2]
        self.h_width = self.train_H.shape[1]
            
        # sample data
        # self.sample_data(train_X.shape[0], val_X.shape[0], test_X.shape[0])
        self.X_train = self.train_X
        self.y_train = self.train_y
        self.X_val = self.val_X
        self.y_val = self.val_y
        self.X_test = self.test_X
        self.y_test = self.test_y
        self.H_train = self.train_H
        self.H_val = self.val_H
        self.H_test = self.test_H
        
        # create original copy value to protect from normalization
        self.X_train_origin = torch.tensor(self.X_train).float()
        self.X_val_origin = torch.tensor(self.X_val).float()
        self.X_test_origin = torch.tensor(self.X_test).float()
        self.y_train_origin = torch.tensor(self.y_train).float()
        self.y_val_origin = torch.tensor(self.y_val).float()
        self.y_test_origin = torch.tensor(self.y_test).float()
        self.H_train_origin = torch.tensor(self.H_train).float()
        self.H_val_origin = torch.tensor(self.H_val).float()
        self.H_test_origin = torch.tensor(self.H_test).float()
        
        # normalization
        self.std_scaler_x = None
        self.std_scaler_y = None
        self.X_scale = None
        self.X_mean = None
        self.y_scale = None
        self.y_mean = None
        
        # claim tensors
        self.X_train_tensor = None
        self.y_train_tensor = None
        self.X_val_tensor = None
        self.y_val_tensor = None
        self.X_test_tensor = None
        self.y_test_tensor = None
        self.X_scale_tensor = None
        self.X_mean_tensor = None
        self.y_scale_tensor = None
        self.y_mean_tensor = None
        self.H_train_tensor = None
        self.H_val_tensor = None
        self.H_test_tensor = None

    def normalization(self, x=True, y=True):
        # save parameters
        self.x_norm = x
        self.y_norm = y
        
        # normalize on features
        if x:
            self.std_scaler_x = StandardScaler()
            self.X_train = self.std_scaler_x.fit_transform(self.X_train)
            self.X_val = self.std_scaler_x.transform(self.X_val)
            self.X_test = self.std_scaler_x.transform(self.X_test)
            self.X_scale = np.array(self.std_scaler_x.scale_)
            self.X_mean = np.array(self.std_scaler_x.mean_)
                
        # normalize on features
        if y:
            self.std_scaler_y = StandardScaler()
            self.y_train = self.std_scaler_y.fit_transform(self.y_train)
            self.y_val = self.std_scaler_y.transform(self.y_val)
            self.y_test = self.std_scaler_y.transform(self.y_test)
            self.y_scale = np.array(self.std_scaler_y.scale_)
            self.y_mean = np.array(self.std_scaler_y.mean_)
    
    def torch_tensor(self, device):
        if device is None:
            device = self.device
        else:
            self.device = device
        
        # get the torch tensor of data and send to the designated device
        self.X_train_tensor = torch.tensor(self.X_train).float()
        self.y_train_tensor = torch.tensor(self.y_train).float()
        self.X_val_tensor = torch.tensor(self.X_val).float()
        self.y_val_tensor = torch.tensor(self.y_val).float()
        self.X_test_tensor = torch.tensor(self.X_test).float()
        self.y_test_tensor = torch.tensor(self.y_test).float()
        self.H_train_tensor = torch.tensor(self.H_train).float()
        self.H_test_tensor = torch.tensor(self.H_test).float()
        self.H_val_tensor = torch.tensor(self.H_val).float()

        self.X_train_origin =  self.X_train_origin
        self.X_val_origin = self.X_val_origin
        self.X_test_origin = self.X_test_origin
        self.y_train_origin = self.y_train_origin
        self.y_val_origin = self.y_val_origin
        self.y_test_origin = self.y_test_origin
        self.H_train_origin = self.H_train_origin
        self.H_val_origin = self.H_val_origin
        self.H_test_origin = self.H_test_origin
            
        if self.x_norm:
            self.X_scale_tensor = torch.tensor(self.X_scale).float()
            self.X_mean_tensor = torch.tensor(self.X_mean).float()
        if self.y_norm:
            self.y_scale_tensor = torch.tensor(self.y_scale).float()
            self.y_mean_tensor = torch.tensor(self.y_mean).float()
                
    def sample_data(self, train_size, val_size, test_size):
        
        # save parameters
        if train_size != 0:
            self.num_train = min(train_size, self.available_train)
        else:
            self.num_train = self.available_train
        if val_size != 0:
            self.num_val = min(val_size, self.available_val)
        else:
            self.num_val = self.available_val
        if test_size != 0:
            self.num_test = min(test_size, self.available_test)
        else:
            self.num_test = self.available_test
        
        self.train_index = np.random.choice(
            self.available_train, 
            self.num_train, 
            replace=False
        )
        
        self.val_index = np.random.choice(
            self.available_val, 
            self.num_val, 
            replace=False
        )
        
        self.test_index = np.random.choice(
            self.available_test, 
            self.num_test, 
            replace=False
        )
        
        self.X_train = self.train_X[self.train_index]
        self.y_train = self.train_y[self.train_index]
        self.X_val = self.val_X[self.val_index]
        self.y_val = self.val_y[self.val_index]
        self.X_test = self.test_X[self.test_index]
        self.y_test = self.test_y[self.test_index]