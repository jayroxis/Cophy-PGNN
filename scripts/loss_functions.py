import torch
from math import sqrt

def inverse_norm(batch, scale, mean):
    return batch * scale + mean

def phy_loss(batchPred, batchReal, batchInput, norm=False):
    num_data = batchPred.size(0)
    if batchInput.dim() == 2:
        H_height = int(sqrt(batchInput.size(1)))
        H_width = H_height
    elif batchInput.dim() == 3:
        H_height = batchInput.size(1)
        H_width = batchInput.size(2)
        
    batchC = batchPred[:, 0: H_width]
    batchEg = batchPred[:, -1]
    loss_phy = torch.sum(
            (
                torch.sum(
                    batchInput.view((-1, H_width, H_height)) * batchC.view((-1, 1, H_width)), 
                    dim=2
                ) - batchC * batchEg.view((-1, 1))
            ) ** 2, 
            dim = 1
        )
    if norm:
        loss_phy /= torch.sum(batchC ** 2, dim=1)
    return loss_phy

def energy_loss(batchPred, batchInput):
    H_height = int(sqrt(batchInput.size(1)))
    H_width = H_height
    batchEg = batchPred[:, -1]
    loss_e = torch.exp(batchEg)
    return loss_e

def mean_field_loss(batchPred, batchReal, batchInput, quadratic=False):
    """ To reduce memory usage, fewer local variables are used here. """
    """ 
        E:   batchPred[:, -1].view((-1, 1))
        Psi: batchPred[:, 0: H_width]
        H:   batchInput.view((-1, H_width, H_height))
        
    """
    H_height = int(sqrt(batchInput.size(1)))
    H_width = H_height
    batchC = batchPred[:, 0: H_width]
    batchEg = batchPred[:, -1]
    batchH = batchInput.view((-1, H_width, H_height))
    
    if quadratic: # eq10
        batchSum = batchC.view((-1, 1, H_width)) @ batchH @ batchC.view((-1, H_width, 1))
        left = batchSum.view(-1) / torch.sum(batchC ** 2, dim=1)
    else:   # eq8
        batchSum = torch.sum(
                        batchH * batchC.view((-1, 1, H_width)), 
                        dim=(1, 2)
                    )
        left =  batchSum / torch.sum(batchC, dim=1)
        
    loss_phy = (left - batchEg) ** 2
    return loss_phy
