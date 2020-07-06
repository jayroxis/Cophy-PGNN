import torch
import math

def inverse_norm(batch, scale, mean):
    return batch * scale + mean

def energy_loss(batchPred, batchInput):
    H_height = int(math.sqrt(batchInput.size(1)))
    H_width = H_height
    batchEg = batchPred[:, -1]
    loss_e = torch.exp(batchEg)
    return loss_e

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

def loss_func(data, loss_list, outputs, e_coff=0.0, s_coff=1.0, batchX=None, batchY=None, batchH=None, norm=False):
    """ 
    Set batchY to None when train on test set. 
    Set batchX to None when only use MSE.
    """

    grad = None
    
    # MSE Loss
    criterion = torch.nn.MSELoss()
    if (batchY is not None) and ('mse_loss' in loss_list):
        loss = criterion(outputs, batchY)
    else:
        loss = 0.0

    if batchH is not None:
        origin_input = batchH
    else:
        # inverse transformation for normalization
        if data.std_scaler_x is not None:
            origin_input = inverse_norm(
                batchX, data.X_scale_tensor, data.X_mean_tensor
            )
        else:
            origin_input = batchX

    if data.std_scaler_y is not None:
        origin_output = inverse_norm(
            outputs, data.y_scale_tensor, data.y_mean_tensor
        )
        origin_y = inverse_norm(
            batchY, data.y_scale_tensor, data.y_mean_tensor
        )
    else:
        origin_output = outputs
        origin_y = batchY

    # physics loss and energy loss
    if 'phy_loss' in loss_list:
        loss_phy = phy_loss(
            origin_output,
            origin_y,
            origin_input,
            norm=norm
        )
    else:
        loss_phy = 0.0

    if 'energy_loss' in loss_list:
        loss_e = energy_loss(
            origin_output,
            origin_input
        )
    else:
        loss_e = 0.0
        
    if type(loss_phy) == torch.Tensor or type(loss_e) == torch.Tensor:
        loss += torch.mean(s_coff * loss_phy + e_coff * loss_e)

    norm_loss_phy = phy_loss(
        origin_output,
        origin_y,
        origin_input,
        norm=True
    )
    norm_loss_phy = torch.mean(norm_loss_phy).item()
    loss_phy = phy_loss(
        origin_output,
        origin_y,
        origin_input,
        norm=False
    )
    loss_phy = torch.mean(loss_phy).item()
    loss_e = energy_loss(
        origin_output,
        origin_input
    )
    loss_e = torch.mean(loss_e).item()
    return loss, loss_phy, norm_loss_phy, loss_e