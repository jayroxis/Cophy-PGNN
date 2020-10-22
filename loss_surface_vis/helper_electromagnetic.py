import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import interp
import pandas as pd

def getFriendlyName(DNN_type):
    name = "NN"
    if DNN_type == "PGNN_":
        name = r'\emph{CoPhy}-PGNN'
    return name

def getBxVsCosineSimilarity(Bx, prediction, target):
    numOfSamples = prediction.shape[0]
    error_per_sample = torch.zeros(numOfSamples, 1)

    error = torch.nn.CosineSimilarity()(prediction, target)
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
        
        y_np = np.asarray(y)
        std_np = np.asarray(std)
        clr = self.getColor(lineID, dashed)

        self.ax.errorbar(x, y_np, std_np, label=legend, fmt='--o', capsize=3, capthick=3)
        self.showLegend()
        
    def showLegend(self):
        self.ax.legend(loc='center right')
