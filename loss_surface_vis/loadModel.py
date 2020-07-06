from DNN import DNN

import glob
import os
import pandas as pd
import torch
import loss_landscapes
import numpy as np

def LoadModel(datasetLoader, model_path, DNN_type, H, Depth, device, initialModel=False):
    D_in  = datasetLoader.x_dim
    D_out = datasetLoader.y_dim
    model_final = DNN(D_in, H, D_out, Depth).to(device)

    fullPath = model_path+DNN_type+"_*/models/*.pt"
    files = sorted(glob.glob(fullPath), key=os.path.getmtime)
    numOfEpochs = len(files) -1
    lastModelFile = files[-1] if (initialModel == False) else files[0]
    if torch.cuda.is_available():
        model_final.load_state_dict(torch.load(lastModelFile))
    else:
        model_final.load_state_dict(torch.load(lastModelFile, map_location='cpu'))

    fullPath = model_path+DNN_type+"_*/results/losses*.csv"
    csv_file = glob.glob(fullPath)[0]
    losses_content = pd.read_csv(csv_file)

    lambda_s_train = losses_content['lambda_s_train']
    lambda_s_test = losses_content['lambda_s_test']
    lambda_e_train = losses_content['lambda_e_train']
    lambda_e_test = losses_content['lambda_e_test']
    
    return (model_final, lambda_s_train, lambda_s_test, lambda_e_train, lambda_e_test)

def loadSurfaceData(DNN_type, surfaceTypeName, model_final, metric, DISTANCE, STEPS, projection_id):
    saved_landscape_array_path = "saved_landscape_array/" + DNN_type+ "/" + str(DISTANCE) + "/" + str(STEPS)+ "/"
    saved_landscape_array_fileName = saved_landscape_array_path + surfaceTypeName + (str(projection_id) if projection_id>0 else "") + ".npy"
    
    loss_data = None
    if not os.path.exists(saved_landscape_array_path):
        os.makedirs(saved_landscape_array_path)
    
    if not os.path.exists(saved_landscape_array_fileName):
        print("Calculating projection: " + saved_landscape_array_fileName)
        loss_data = loss_landscapes.random_plane(model_final, metric, distance=1/DISTANCE, steps=STEPS, normalization='filter', deepcopy_model=True)
        np.save(saved_landscape_array_fileName, loss_data)
    else:
        print("Loading projection from disk: " + saved_landscape_array_fileName)
        loss_data = np.load(saved_landscape_array_fileName)
        
    return loss_data