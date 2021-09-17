import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch.nn.functional as f

import sys
import os
sys.path.append(os.path.abspath('../loss_surface_vis'))
sys.path.append(os.path.abspath('../scripts'))
from loss_functions_electromagnetic import multiply_Eg_C

# Calculates | HC - EC |/ |HC|
def getEigError(eigenVector, eigenval, H):
    eigenVector_shape_half = int(eigenVector.shape[1]/2)
    batchC = eigenVector.view(-1, eigenVector.shape[1], 1)
    batchEg = eigenval.view(-1, 2, 1)
    HC = torch.matmul(H, batchC)
    EC = multiply_Eg_C(batchEg, batchC)
    EC = multiply_Eg_C(batchEg, EC) 
    EC = EC*2*2
    EC = EC*(torch.tensor(np.pi))
    EC = EC*(torch.tensor(np.pi))# This assumes lam0=1
    Total = (HC - EC)**2
    loss_phy = torch.sqrt(torch.sum(Total, dim=1))
    loss_phy /= torch.sqrt(torch.sum(HC ** 2, dim=1))
    
    return loss_phy

def getCosSimMeasure(prediction, target):
    return torch.nn.CosineSimilarity()(prediction, target)

def getMSE(prediction, target):
    # scale the vectors so that the first element of both is equal
    scale = torch.div(prediction[:, 0], target[:, 0]).view(prediction.shape[0], -1)
    prediction_v = torch.div(prediction, scale)

    # Calculate error
    num_of_examples = target.shape[0]
    error=torch.zeros(num_of_examples)
    for i in range(num_of_examples):
        error[i] = torch.nn.MSELoss()(prediction_v[i, :], target[i, :])
    return error

def getOverlapIntegral(prediction, target, useRealComp=False):
    # Convert to numpy complex format
    shape_half = int(target.shape[1]/2)
    prediction2 = prediction[:,:shape_half].cpu().detach().numpy() + 1j * prediction[:,shape_half:].cpu().detach().numpy()
    target2 = target[:,:shape_half].cpu().detach().numpy() + 1j * target[:,shape_half:].cpu().detach().numpy()
    
    # Calculate error
    num_of_examples = prediction2.shape[0]
    error=torch.zeros(num_of_examples)
    for i in range(num_of_examples):
        error1 = np.vdot(prediction2[i, :], target2[i, :]) # vdot takes care of conjugating automatically.
        if useRealComp:
            error1 = np.real(error1)
        else: 
            error1 = np.absolute(error1)
        error[i] = torch.tensor(error1)
    return error


def getBxVsCosineSimilarity(Bx, batchInput, prediction, target, measure="Cos"):
    numOfSamples = prediction.shape[0]
    error_per_sample = torch.zeros(numOfSamples, 1)
    
    # Normalize vectors
    prediction_v = prediction[:, :-2]
    prediction_v = f.normalize(prediction_v, p=2, dim=-1)
    target_v = target[:, :-2]
    target_v = f.normalize(target_v, p=2, dim=-1)

    if measure=="Cos":
        error = getCosSimMeasure(prediction_v, target_v)
    elif measure=="OverInt":
        error = getOverlapIntegral(prediction_v, target_v)
    elif measure=="OverIntRealVal":
        error = getOverlapIntegral(prediction_v, target_v, True)
    elif measure=="MSE":
        error = getMSE(prediction_v, target_v)
    elif measure=="EigenError":
        error = getEigError(prediction_v, prediction[:, -2:], batchInput).view(-1)
    elif measure=="EigenError2":
        error = getEigError(prediction_v, target[:, -2:], batchInput).view(-1)
    else:
        raise
    
    error_per_sample[:, 0] = error

    Bx_inds = torch.argsort(Bx, dim=0)
    Bx_sorted = Bx[Bx_inds].view(Bx.shape[0], -1)

    error_per_sample_sorted = error_per_sample[Bx_inds].view(error_per_sample.shape)
    
    return torch.cat((Bx_sorted, error_per_sample_sorted), dim=1)

def getFittedData(Bx_vs_cosineSimilarity, x_range):
    x_axis = (Bx_vs_cosineSimilarity)[:, 0].t().cpu().detach().numpy()
    y_axis = (Bx_vs_cosineSimilarity)[:, 1].t().cpu().detach().numpy()
    return np.transpose(y_axis)

def GetStatistics(Bx, Bx_vs_cosineSimilarity_list):
    Bx_vs_cosineSimilarity_array = np.hstack(Bx_vs_cosineSimilarity_list).reshape(1, -1)
    Bx_ = np.tile(Bx,(1,len(Bx_vs_cosineSimilarity_list)))
    Bx_vs_cosineSimilarity_array = np.concatenate((Bx_, Bx_vs_cosineSimilarity_array), axis=0)
    df = pd.DataFrame(Bx_vs_cosineSimilarity_array.T, columns=[ 'Bx','cosineSimilarity'])
    df2 = df.groupby('Bx').mean().reset_index()
    df3 = df.groupby('Bx').std().reset_index()
    Bx_vs_cosineSimilarity_mean = df2['cosineSimilarity'].values.tolist()
    Bx_vs_cosineSimilarity_std = df3['cosineSimilarity'].values.tolist()
    return (Bx_vs_cosineSimilarity_mean, Bx_vs_cosineSimilarity_std)

# Plot helper
class PlotHelper():
    def __init__(self, title, xLabel, yLabel, xticks, line_location):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel(xLabel)
        self.ax.set_ylabel(yLabel)
        self.ax.set_xticks(xticks)
        self.colors = ['g', 'b', 'c', 'r', 'm', 'y', 'k']
        self.ax.axvline(x=line_location,dashes=(5,5),color="black", lw=0.5)
        
    def getColor(self, lineID, dashed):
        clr = ''
        if lineID != -1:
            clr = self.colors[lineID % len(self.colors)]
        if dashed == True:
            clr = clr + '--'  
        return clr
        
    def updatePlotXYSTD(self, x, y, std, legend, lineID=-1, dashed=False):
        legend = legend.replace('BB', 'NN')
        if legend == 'CoPhy':
            legend = r'\emph{CoPhy}-PGNN'
        if legend == 'Analogue':
            legend = r'PGNN-\emph{analogue}'
        
        y_np = np.asarray(y)
        std_np = np.asarray(std)
        clr = self.getColor(lineID, dashed)

        self.ax.errorbar(x, y_np, std_np, label=legend, fmt='--o', capsize=3, capthick=3)
        self.showLegend()
        
    def showLegend(self):
        self.ax.legend(loc='upper right')
