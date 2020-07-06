import numpy as np
import pandas as pd
import glob

from parse_results import parse_results
from presets import LoadFilter

class Filter(object):
    def __init__(self, path):
        self.path=path
        self.dicts = []
    
    def match(self, mode):
        list_of_files = glob.glob(self.path)
        overlap = []
        epoch = []
        name = []
        mode_list = []
        loss_list = []
        loss_csv = {}
        train_size = []
        file_path = []
        if len(list_of_files) == 0:
            raise RuntimeError("Empty repository, no txt files found.")
        
        model_filter = LoadFilter()
        for i in range(len(list_of_files)):
            file = list_of_files[i]
            d = parse_results(file)
            
            loss_params = d['params']['loss_params']
            if getattr(model_filter, 'NSE_DNNex')(d) and loss_params['cold_start']:
                if loss_params['cold_start']['mode'] == mode:
                    mode_list.append(mode)
                    self.dicts.append(d)
                    name.append(d['name'])
                    size = d['params']['data_params']['train_size']
                    train_size.append(size)
                    overlap.append(d['wave_function_overlap'])
                    stop_epoch = min(int(d['epoch']), 499)
                    epoch.append(stop_epoch)
                    df = pd.read_csv(d['loss_csv'])
                    if size not in loss_csv:
                        loss_csv[size] = []
                    loss_csv[size].append(df)
                    df = df.iloc[[stop_epoch]]
                    loss_list.append(df)
                    file_path.append(file)
                

        overlap = np.array(overlap)
        epoch = np.array(epoch)
        train_size = np.array(train_size)
        loss_df = pd.concat(loss_list, sort=False)
        
        df = pd.DataFrame({
            'names': name,
            'mode': np.array(mode_list),
            'path': np.array(file_path),
            'overlap': overlap,
            'epoch': epoch,
            'train size': train_size
        }).reset_index(drop=True)
        
        self.df = df.join(loss_df.reset_index(drop=True))
        self.df = self.df.rename(columns={'test_mse':'mse'})
        print("Number of %s matches = %d" % (mode, len(self.dicts)))
        return loss_csv