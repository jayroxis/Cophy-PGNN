import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import interp

def getFriendlyName(DNN_type):
    name = "NN"
    if DNN_type == "PGNN_OnlyDTr":
        name = r'\emph{CoPhy}-PGNN (only-$\mathcal{D}_{Tr}$)'
    elif DNN_type == "PGNN_LF":
        name = r'\emph{CoPhy}-PGNN (Label-free)'
    elif DNN_type == "PGNN_":
        name = r'\emph{CoPhy}-PGNN'
    return name

def getBxVsCosineSimilarity(Bx, prediction, target):
    numOfSamples = prediction.shape[0]
    error_per_sample = torch.zeros(numOfSamples, 1)

    error = torch.nn.CosineSimilarity()(prediction[:, :-1], target[:, :-1])
    error_per_sample[:, 0] = error

    Bx_inds = torch.argsort(Bx, dim=0)
    Bx_sorted = Bx[Bx_inds].view(Bx.shape[0], -1)

    error_per_sample_sorted = error_per_sample[Bx_inds].view(error_per_sample.shape)

    return torch.cat((Bx_sorted, error_per_sample_sorted), dim=1)

def getFittedData(Bx_vs_cosineSimilarity, x_range):
    x_axis = (Bx_vs_cosineSimilarity)[:, 0].t().cpu().detach().numpy()
    y_axis = (Bx_vs_cosineSimilarity)[:, 1].t().cpu().detach().numpy()
    Bx_vs_cosineSimilarit_y = interp(x_range, x_axis, y_axis)
    return np.transpose(Bx_vs_cosineSimilarit_y)

def GetStatistics(Bx_vs_cosineSimilarity_list):
    Bx_vs_cosineSimilarity_array = np.vstack(Bx_vs_cosineSimilarity_list)
    Bx_vs_cosineSimilarity_mean = np.mean(Bx_vs_cosineSimilarity_array, axis=0)
    Bx_vs_cosineSimilarity_std = np.std(Bx_vs_cosineSimilarity_array, axis=0)
    return (Bx_vs_cosineSimilarity_mean, Bx_vs_cosineSimilarity_std)

# Plot helper
class PlotHelper():
    def __init__(self, title, xLabel, yLabel):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel(xLabel)
        self.ax.set_ylabel(yLabel)
        self.colors = ['g', 'b', 'c', 'r', 'm', 'y', 'k']
        self.ax.axvline(x=0.5,dashes=(5,5),color="black", lw=0.5)

    def getColor(self, lineID, dashed):
        clr = ''
        if lineID != -1:
            clr = self.colors[lineID % len(self.colors)]
        if dashed == True:
            clr = clr + '--'  
        return clr

    def updatePlotYSTD(self, y, std, legend, lineID=-1, dashed=False):
        x = np.arange(len(y))
        y_np = np.asarray(y)
        std_np = np.asarray(std)
        clr = self.getColor(lineID, dashed)

        self.ax.plot(x, y_np, clr, label=legend)
        self.ax.fill_between(x,y_np - std_np, y_np + std_np, color=clr[0], alpha=0.2)
        self.showLegend()

    def updatePlotXYSTD(self, x, y, std, legend, lineID=-1, dashed=False):
        y_np = np.asarray(y)
        std_np = np.asarray(std)
        clr = self.getColor(lineID, dashed)

        self.ax.plot(x, y_np, clr, label=legend)
        self.ax.fill_between(x,y_np - std_np, y_np + std_np, color=clr[0], alpha=0.2)
        self.showLegend()
    
    def updatePlotXY(self, x, y, legend, lineID=-1, dashed=False):
        clr = self.getColor(lineID, dashed)
        self.ax.plot(x.cpu().detach().numpy(), y.cpu().detach().numpy(), clr, label=legend)
        self.showLegend()

    def updatePlotY(self, y, legend, lineID=-1, dashed=False):
        clr = self.getColor(lineID, dashed)
        self.ax.plot(y, clr, label=legend)
        self.showLegend()

    def showLegend(self):
        self.ax.legend(loc='lower right')
