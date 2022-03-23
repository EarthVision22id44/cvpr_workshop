import torch

def bicubic_interp(tensor,scale_factor):
    
    interp = torch.nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)
    ssh_bicb = interp(tensor.unsqueeze(0).unsqueeze(0))
    
    return ssh_bicb[0,0,:,:]