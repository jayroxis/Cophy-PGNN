import torch
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
        
def grad_cosine(grad_1, grad_2):
    cos = np.zeros(len(grad_1))
    for i in range(len(grad_1)):
        cos_arr = grad_1[i] * grad_2[i]
        cos_arr /= np.sqrt(np.sum(grad_1[i] ** 2))
        cos_arr /= np.sqrt(np.sum(grad_2[i] ** 2))
        cos[i] = np.sum(cos_arr)
    return cos

def grad_vs_optimal(grad_list, param_list):
    final_param = param_list[-1]
    cos = []
    for i in range(len(param_list) - 1):
        param = param_list[i]
        grad = grad_list[i]
        ideal_direction = [param[j] - final_param[j] for j in range(len(param))]
        cos.append(grad_cosine(grad, ideal_direction))
    return np.stack(cos)

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                plt.Line2D([0], [0], color="b", lw=4),
                plt.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

class GradAnalysis(object):
    def __init__(self, model):
        self.model = model
        self.names = []
        self.params = []
        self.grad = []
        
        self.get_param()
        
    def get_param(self):
        self.params = []
        for n, p in self.model.named_parameters():
            if (p.requires_grad) and ("bias" not in n):
                self.names.append(n)
                self.params.append(p.data.clone().cpu().numpy())
        return self.params
                
    def loss_grad(self, loss):
        # Backward and optimize   
        loss.backward(retain_graph=True)
        self.grad = [
            p.grad.clone().cpu().numpy() 
            for n, p in self.model.named_parameters() 
            if (p.requires_grad) and ("bias" not in n)
        ]
        return self.grad
        
    def clear_grad(self):
        for n, p in self.model.named_parameters():
            if (p.requires_grad) and ("bias" not in n):
                p.grad.data.zero_()