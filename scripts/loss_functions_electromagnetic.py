import torch
from math import sqrt
import numpy as np

# The batch of this file is vectors of the form [Real Imaginary].
# Real takes the first half of the vector, and Imaginary takes the second half.
# Last two elements are eignevalue (real, img)

def inverse_norm(batch, scale, mean):
    return batch * scale + mean

# Multiplies a vector with a complex number in the adopted representation.
def multiply_Eg_C(batchEg, batchC):
    Eg_real = batchEg[:, 0].view(-1, 1, 1)
    Eg_img = batchEg[:, 1].view(-1, 1, 1)
    
    l = int(batchC.shape[1]/2)

    Eg_real_times_batchC = Eg_real*batchC
    Eg_img_times_batchC = Eg_img*batchC
    Eg_img_times_batchC = torch.cat([Eg_img_times_batchC[:, l:], Eg_img_times_batchC[:, :l]], dim=1)
    Eg_img_times_batchC[:, :l] = -Eg_img_times_batchC[:, :l]

    return Eg_real_times_batchC + Eg_img_times_batchC

def phy_loss(batchPred, batchReal, batchInput, norm=False, HC_enabled=False):
    num_data = batchPred.size(0)
    if batchInput.dim() == 2:
        H_height = int(sqrt(batchInput.size(1)))
        H_width = H_height
    elif batchInput.dim() == 3:
        H_height = batchInput.size(1)
        H_width = batchInput.size(2)
        
    batchC = batchPred[:, :-2].view(-1, H_height, 1)
    batchEg = batchPred[:, -2:].view(-1, 2, 1)

    HC = torch.matmul(batchInput, batchC)
    EC = multiply_Eg_C(batchEg, multiply_Eg_C(batchEg, batchC))*2*2*torch.tensor(np.pi).cuda()*torch.tensor(np.pi).cuda()# This assumes lam0=1
    Total = (HC - EC)**2

    loss_phy = torch.sum(Total, dim=1)

    if norm:
        loss_phy /= torch.sum((HC ** 2) if HC_enabled == True else (batchC ** 2), dim=1)
    return loss_phy



def overlap_squared_loss(prediction_, target_):
    prediction = prediction_[:, :-2]
    target = target_[:, :-2]
    shape_half = int(target.shape[1]/2)
    
    prediction1 = prediction[:,:shape_half].view(-1, shape_half)
    prediction2 = prediction[:,shape_half:].view(-1, shape_half)
    target1 = target[:,:shape_half].view(-1, shape_half)
    target2 = -target[:,shape_half:].view(-1, shape_half)
    
    a = torch.sum(prediction1* target1, dim=1)
    b = torch.sum(prediction2* target2, dim=1)
    c = torch.sum(prediction1* target2, dim=1)
    d = torch.sum(target1* prediction2, dim=1)

    return -((a - b)**2 + (c + d)**2)


def energy_loss(batchPred, batchInput):
    batchEg = batchPred[:, -2:]
    loss_e = torch.exp(-batchEg[:, 0])
    return loss_e
