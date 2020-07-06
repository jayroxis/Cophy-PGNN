
import torch
import numpy as np
from torch.autograd import Variable

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import os

# restore hamiltonian from nonzeros
def restore_h(nonzero, nonzero_loc, dim):
    data_count = nonzero.shape[0]
    H = torch.zeros((data_count, dim, dim), device=nonzero.device).float()
    triangle_shape = (data_count, int(dim * (dim + 1) / 2))
    upper_triangle = torch.zeros(triangle_shape, device=nonzero.device)
    upper_triangle[:, nonzero_loc] = nonzero
    iu1 = np.triu_indices(dim)
    il1 = np.tril_indices(dim)
    H[:, iu1[0], iu1[1]] = upper_triangle
    H_t = torch.transpose(H, dim0=1, dim1=2)
    diag = torch.diagonal(H, offset=0, dim1=1, dim2=2)
    diag_matrix = torch.diag_embed(diag, offset=0, dim1=1, dim2=2)
    H += H_t - diag_matrix
    return H


def getData(n_sites, path_to_data, train_test, dataset='new'):
    if dataset == 'new':
        try:
             # load data from server
            directory = path_to_data + "%d-qubits/100k-instances/single-phase/%s/" % (n_sites, train_test)

            # read the files
            with open(directory + "header.bin", 'rb') as file:
                header = np.fromfile(file, dtype=np.int32)
            with open(directory + "matloc.bin", 'rb') as file:
                matloc = np.fromfile(file, dtype=np.int32)
            with open(directory + "matval.bin", 'rb') as file:
                matval = np.fromfile(file, dtype=np.float64)
            with open(directory + "wavefx.bin", 'rb') as file:
                wavefx = np.fromfile(file, dtype=np.float64)
            with open(directory + "energy.bin", 'rb') as file:
                energy = np.fromfile(file, dtype=np.float64)
            with open(directory + "fields.bin", 'rb') as file:
                fields = np.fromfile(file, dtype=np.float64)
            with open(directory + "szdiag.bin", 'rb') as file:
                szdiag = np.fromfile(file, dtype=np.float64)

            # reshape/format data
            data_count = header[0]
            dim = header[2]
            energy = torch.tensor(energy).float().view(data_count, 1)
            Bx = torch.tensor(fields).float().view(data_count, 1)
            
            szdiag = torch.diag(torch.tensor(szdiag)).float()
            psi = torch.tensor(wavefx).float().view(data_count, dim)
            
            nonzero_H_Ids = matloc.reshape((data_count, -1))
            nonzero_H = torch.tensor(matval).float().view(data_count, -1)

        except Exception as e:
            print(e)

        return Bx, nonzero_H, psi, energy, data_count, nonzero_H_Ids, szdiag
    elif dataset == 'old':
        """
        Compile all the data into arrays for use in NN training ans testing.

        Parameters:
            n_sites (int): The number of sites in your model
            path_to_data (str): The path to the directory where the data is.
            train_test (str): Options 'train' or 'test' depending on desired dataset.

        Returns:
            BxBz (np.ndarray): An array with the ordered Bx, Bz pairs in each row.
            Ham (np.ndarray): An array with the flattened input Hamiltonian in each row.
            targets (np.ndarray): An array with the ground state wavefunction in each row.
            eigenvalues (np.ndarray): An array with the energy eigenvalues for the ground state in each row.
            datapoints (int): The number of datapoints loaded.
        """

        with open('%s%s-site-field-gs-%s.bin' % (path_to_data, n_sites, train_test), 'rb') as f:
            params1 = np.fromfile(f, dtype=np.int32, count=3)
            data1 = np.fromfile(f, dtype=np.float64)
        BxBz = data1[0:params1[0] * params1[1]]
        targets = data1[params1[0] * params1[1]::]
        BxBz = BxBz.reshape((params1[0], params1[1]))
        targets = targets.reshape((params1[0], params1[2]))

        with open('%s%s-site-hamil-gs-%s.bin' % (path_to_data, n_sites, train_test), 'rb') as f:
            params1 = np.fromfile(f, dtype=np.int32, count=3)
            data1 = np.fromfile(f, dtype=np.float64)
        Ham = data1[0:params1[0] * params1[1]]
        Ham = Ham.reshape((params1[0], params1[1]))

        with open('%s%s-site-gs-prior-%s.bin' % (path_to_data, n_sites, train_test), 'rb') as f:
            params1 = np.fromfile(f, dtype=np.int32, count=3)
            data1 = np.fromfile(f, dtype=np.float64)
        eigenvals = data1[params1[0] * (params1[1]**2)::]
        eigenvals = eigenvals.reshape((params1[0], params1[1] * params1[2]))
        eigenvals = eigenvals[:,0]
        eigenvals = eigenvals.reshape((params1[0], 1))
        datapoints = params1[0]

        return [BxBz, Ham, targets, eigenvals, datapoints]


# ------------> data loader start here <----------------
class DatasetLoader(object):
    def __init__(
        self, 
        data_path, 
        n_sites, 
        train_size, 
        val_size, 
        test_size,
        dataset='new'
    ):
        self.dataset = dataset
        
        # store parameters
        self.data_path = data_path
        self.n_site = n_sites
        if dataset == 'old':
            # load training data
            training_set = getData(
                n_sites=n_sites,
                path_to_data=data_path,
                train_test='train',
                dataset=dataset
            )
            
            # load test data
            test_set = getData(
                n_sites=n_sites,
                path_to_data=data_path,
                train_test='test',
                dataset=dataset
            )
        else:
            # load training data
            training_set = getData(
                n_sites=n_sites,
                path_to_data=data_path,
                train_test='training',
                dataset=dataset
            )
            
            # load test data
            test_set = getData(
                n_sites=n_sites,
                path_to_data=data_path,
                train_test='testing',
                dataset=dataset
            )
            val_set = getData(
                n_sites=n_sites,
                path_to_data=data_path,
                train_test='validation',
                dataset=dataset
            )
            
            # interpret training variables
            BxBz_val = val_set[0]
            Ham_val = val_set[1]
            targets_val = val_set[2]
            eigenvals_val = val_set[3]
            
            # phased-locked validation dataset: always positive psi_0
            self.val_X = Ham_val
            self.val_y = np.concatenate((targets_val, eigenvals_val),axis=1)
            self.val_y = torch.tensor(self.val_y).float()
            self.available_val = Ham_val.shape[0]
            self.sign_val = torch.sign(self.val_y[:, 0]).view(-1, 1)
            self.val_y[:, :-1] *= self.sign_val
            
            # szdiag
            self.szdiag_train = training_set[6]
            self.szdiag_val = val_set[6]
            self.szdiag_test = test_set[6]
            
            # original tensors
            self.nonzero_loc = training_set[5][0]
        
        # interpret training variables
        BxBz_train = training_set[0]
        Ham_train = training_set[1]
        targets_train = training_set[2]
        eigenvals_train = training_set[3]
        
        # interpret test variables
        BxBz_test = test_set[0]
        Ham_test = test_set[1]
        targets_test = test_set[2]
        eigenvals_test = test_set[3]

        # phased-locked training dataset: always positive psi_0
        self.train_X = Ham_train
        self.train_y = np.concatenate((targets_train, eigenvals_train),axis=1)
        self.train_y = torch.tensor(self.train_y).float()
        self.available_train = Ham_train.shape[0]
        self.sign_train = torch.sign(self.train_y[:, 0]).view(-1, 1)
        self.train_y[:, :-1] *= self.sign_train
        
        # phased-locked test dataset: always positive psi_0
        self.test_X = Ham_test
        self.test_y = np.concatenate((targets_test, eigenvals_test),axis=1)
        self.test_y = torch.tensor(self.test_y).float()
        self.available_test = Ham_test.shape[0]
        self.sign_test = torch.sign(self.test_y[:, 0]).view(-1, 1)
        self.test_y[:, :-1] *= self.sign_test
       
        # dimensions
        self.x_dim = self.train_X.shape[1]
        self.y_dim = self.train_y.shape[1]
        self.h_height = n_sites * n_sites
        self.h_width = n_sites * n_sites
            
        # sample data
        if dataset == 'new':
            self.sample_data(train_size, val_size, test_size)
        else:
            self.sample_data_old(train_size, val_size, test_size)
        
        # create original copy value to protect from normalization
        self.X_train_origin = torch.tensor(self.X_train).float()
        self.X_val_origin = torch.tensor(self.X_val).float()
        self.X_test_origin = torch.tensor(self.X_test).float()
        self.y_train_origin = torch.tensor(self.y_train).float()
        self.y_val_origin = torch.tensor(self.y_val).float()
        self.y_test_origin = torch.tensor(self.y_test).float()
        
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
        self.X_train_tensor = torch.tensor(self.X_train).float().to(device)
        self.y_train_tensor = torch.tensor(self.y_train).float().to(device)
        self.X_val_tensor = torch.tensor(self.X_val).float().to(device)
        self.y_val_tensor = torch.tensor(self.y_val).float().to(device)
        self.X_test_tensor = torch.tensor(self.X_test).float().to(device)
        self.y_test_tensor = torch.tensor(self.y_test).float().to(device)

        if self.dataset == 'new':
            self.X_train_origin = restore_h(
                self.X_train_origin.to(device), self.nonzero_loc, self.h_height
            )
            self.X_val_origin = restore_h(
                self.X_val_origin.to(device), self.nonzero_loc, self.h_height
            )
            self.X_test_origin = restore_h(
                self.X_test_origin.to(device), self.nonzero_loc, self.h_height
            )
            self.y_train_origin = self.y_train_origin.to(device)
            self.y_val_origin = self.y_val_origin.to(device)
            self.y_test_origin = self.y_test_origin.to(device)
            
        if self.x_norm:
            self.X_scale_tensor = torch.tensor(self.X_scale).float().to(device)
            self.X_mean_tensor = torch.tensor(self.X_mean).float().to(device)
        if self.y_norm:
            self.y_scale_tensor = torch.tensor(self.y_scale).float().to(device)
            self.y_mean_tensor = torch.tensor(self.y_mean).float().to(device)
                
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

    def sample_data_old(self, train_size, val_size, test_size):
        
        # save parameters
        self.num_train = train_size
        self.num_val = val_size
        self.num_test = test_size
        
        # get random indexes first
        if self.num_train == 0 and self.num_val == 0:
            self.num_train = self.available_train
            self.X_train = self.train_X.copy()
            self.y_train = self.train_y.copy()
            
            self.num_val = self.available_train
            self.X_val = self.train_X.copy()
            self.y_val = self.train_X.copy()
        else:
            # sampling by indexes
            # if any size set to zero, then use entire dataset
            train_val_index = np.random.choice(
                self.available_train, 
                self.num_train + self.num_val, 
                replace=False
            )
            
            if self.num_train:
                self.train_index = train_val_index[:self.num_train]
                self.X_train = self.train_X[self.train_index]
                self.y_train = self.train_y[self.train_index]

            if self.num_val:    
                self.val_index = train_val_index[self.num_train:]
                self.X_val = self.train_X[self.val_index]
                self.y_val = self.train_y[self.val_index]
        
        if self.num_test == 0:
            self.num_test = self.available_test
            self.X_test = self.test_X
            self.y_test = self.test_y
        else:
            self.test_index = np.random.choice(
                self.available_test, 
                self.num_test, 
                replace=False
            )
            self.X_test = self.test_X[self.test_index]
            self.y_test = self.test_y[self.test_index]            
        