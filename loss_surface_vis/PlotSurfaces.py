import matplotlib.pyplot as plt
import os
import numpy as np
import loss_landscapes

from helper import getFriendlyName
from loadModel import loadSurfaceData


class PlotLinearInterpolation():
    def __init__(self, initialModel, models):
        self.initialModel = initialModel
        self.models = models
        
    def plotLinearInterpolation(self, metric, title, id, axs, STEPS):    
        loss_data_fin2 = loss_landscapes.linear_interpolation(self.initialModel, self.models[0][1], metric, steps=STEPS, deepcopy_model=True)
        loss_data_fin3 = loss_landscapes.linear_interpolation(self.initialModel, self.models[1][1], metric, steps=STEPS, deepcopy_model=True)

        axs[id].plot(loss_data_fin2, label=getFriendlyName(self.models[0][0]))
        axs[id].plot(loss_data_fin3, label=getFriendlyName(self.models[1][0]))
        axs[id].set_ylabel(title)

        axs[id].legend(loc='upper right', fontsize=14)


class PlotSurfaces():
    def __init__(self, DISTANCES, STEPS, figuresPath):
        self.DISTANCES = DISTANCES
        self.STEPS = STEPS
        self.figuresPath = figuresPath
        
    def plotSurfaces(self, models, DNN_types, metrics, surfaceTypeName="Generic"):
        numOfModels = len(DNN_types)
        
        f, axes = plt.subplots(len(self.DISTANCES), numOfModels, figsize=(18, 3), dpi= 300,)
        
        # For different scales
        for i in range(len(self.DISTANCES)):
            loss_data_fins = []
            DISTANCE = self.DISTANCES[i]
            X = np.array([(k - int(self.STEPS/2))*DISTANCE for k in range(self.STEPS)])
            Y = X
            
            # Get surface data for all models
            for j in range(numOfModels):
                loss_data_fin = loadSurfaceData(DNN_types[j], surfaceTypeName, models[j], metrics[j], DISTANCE, self.STEPS, 0)
                loss_data_fins.append(loss_data_fin)

            # Plot surface data for all models, normalized across all models.
            for j in range(numOfModels):    
                loss_data_fin = loss_data_fins[j]
                ax = plt.subplot(len(self.DISTANCES), numOfModels, j+numOfModels*i+1)
                contours = plt.contour(X, Y, loss_data_fin, 15, colors='black') # 
                plt.clabel(contours, inline=True, fontsize=10)
                plt.imshow(loss_data_fin, extent=[X[0], X[-1], Y[0], Y[-1]], origin='lower') # , cmap='RdGy', alpha=0.5
                maxVal = np.amax([np.amax(l) for l in loss_data_fins])
                minVal = np.amin([np.amin(l) for l in loss_data_fins])
                plt.pcolor(X, Y, loss_data_fin, vmin=minVal, vmax=maxVal, cmap='Reds')
                plt.colorbar()
 
                name = getFriendlyName(DNN_types[j]) + " / " + surfaceTypeName
                
                ax.title.set_text(name)
                plt.plot([0], [0], marker='o', markersize=4, color="blue")


        f.show()
        
        if not os.path.exists(self.figuresPath):
            os.makedirs(self.figuresPath)   
        f.savefig(os.path.join(self.figuresPath, surfaceTypeName+".pdf"), bbox_inches='tight')