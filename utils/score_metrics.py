import numpy as np
import torch
from math import pi

from torch.nn import _reduction as _Reduction
from torch.nn.functional import conv2d

# ETA
def RMSE(img, ref):
    
    rmse=(((img-ref)**2).mean()).sqrt()
    
    return rmse.item()

def _fspecial_gaussian(size, channel, sigma):
    coords = torch.tensor([(x - (size - 1.) / 2.) for x in range(size)])
    coords = -coords ** 2 / (2. * sigma ** 2)
    grid = coords.view(1, -1) + coords.view(-1, 1)
    grid = grid.view(1, -1)
    grid = grid.softmax(-1)
    kernel = grid.view(1, 1, size, size)
    kernel = kernel.expand(channel, 1, size, size).contiguous()
    return kernel

def SSIM(inpt, target, max_val=1, k1=0.01, k2=0.03, channel=1):
    
    inpt=inpt.unsqueeze(0).unsqueeze(0)
    target = target.unsqueeze(0).unsqueeze(0)
    kernel=_fspecial_gaussian(size=11, channel=1, sigma=1.5)
    
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    mu1 = conv2d(inpt, kernel, groups=channel)
    mu2 = conv2d(target, kernel, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(inpt * inpt, kernel, groups=channel) - mu1_sq
    sigma2_sq = conv2d(target * target, kernel, groups=channel) - mu2_sq
    sigma12 = conv2d(inpt * target, kernel, groups=channel) - mu1_mu2

    v1 = 2 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2

    ssim = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)
    
    return ssim.mean().item()

#W

# Average Endpoint error
def EPE(w_assim,w_estimate):
    
    u=w_assim[0,:,:]
    v=w_assim[1,:,:]
    
    uhat=w_estimate[0,:,:]
    vhat=w_estimate[1,:,:]
    
    norm_error = torch.sqrt((u-uhat)**2 + (v-vhat)**2)
    epe=norm_error.mean()
    
    return epe

# Average Angular Error
def AAE(w_assim,w_estimate):
    
    u=w_assim[0,:,:]
    v=w_assim[1,:,:]
    
    uhat=w_estimate[0,:,:]
    vhat=w_estimate[1,:,:]
    
    dot_product = u*uhat + v*vhat
    norm_w = torch.sqrt(u**2 + v**2)
    norm_what= torch.sqrt(uhat**2 + vhat**2)
    
    #preventing round error with clamp avoiding Nan in acos
    cos = dot_product/(norm_w*norm_what)
    cos = torch.clamp(cos,max=1,min=-1)

    angular_error = (180/pi)*torch.acos(cos)
    
    # filtering nan and inf
    angular_error[angular_error == float("inf")] = float("nan")
    angular_error=angular_error[angular_error==angular_error]
    aae=angular_error.mean()
    
    return aae



######################################## SMOOTHNESS STATS ###

# grad w norma
def norm_gradw(w,dx=1,dy=1):
    
    u = w[0,:,:]
    v = w[1,:,:]
    
    grad_ux, grad_uy = grad_mat(u,dx,dy)
    grad_vx, grad_vy = grad_mat(v,dx,dy)
    
    norm_grad = torch.sqrt(torch.sum(grad_ux**2) + torch.sum(grad_uy**2) + 
                 torch.sum(grad_vx**2) +torch.sum(grad_vy**2))
    
    return norm_grad

# div w norma
def norm_divw(w,dx=1,dy=1):
    
    u = w[0,:,:]
    v = w[1,:,:]
    
    div_w = div_mat(u,v,dy,dx)
    norm_div = torch.sqrt(torch.sum(div_w**2))
    
    return norm_div

# lap w norma
def norm_lapw(w,dx=1,dy=1):
    
    u = w[0,:,:]
    v = w[1,:,:]
    
    grad_ux, gradu_uy = grad_mat(u,dx,dy)
    grad_vx, grad_vy = grad_mat(v,dx,dy)
    
    lap_u = div_mat(grad_ux,gradu_uy,dy,dx)
    lap_v = div_mat(grad_vx,grad_vy,dy,dx)
    
    norm_lap = torch.sqrt(torch.sum(lap_u**2)+torch.sum(lap_v**2))
    
    return norm_lap

### Gradient, Divergence, Laplacian ###

def grad_mat(M,dx,dy):
    #cut the border automatically
    # M should have the padded border
    
    M_ij = M[1:(M.shape[0]-1),1:(M.shape[1]-1)]
    M_ip1j = M[2:M.shape[0],1:(M.shape[1]-1)]
    M_ijp1 = M[1:(M.shape[0]-1),2:M.shape[1]]
    
    #grad = (M_ip1j - M_ij)/dx + (M_ijp1 - M_ij)/dy
    grad_x, grad_y = (M_ip1j - M_ij)/dx , (M_ijp1 - M_ij)/dy

    # return arrays without border
    return grad_x, grad_y

def div_mat(U,V,dx,dy):
    #cut the border automatically
    # U,V should have the padded border
    
    U_ij = U[1:(U.shape[0]-1),1:(U.shape[1]-1)]
    V_ij = V[1:(V.shape[0]-1),1:(V.shape[1]-1)]
    
    U_ip1j = U[2:U.shape[0],1:(U.shape[1]-1)]
    V_ijp1 = V[1:(V.shape[0]-1),2:V.shape[1]]
    
    div = (U_ip1j - U_ij)/dx + (V_ijp1-V_ij)/dy
    
    return div

def lap_mat(M,dx,dy):
    
    grad_Mx, grad_My = grad_mat(M,dx,dy)
    lap_M = div_mat(grad_Mx,grad_My,dx,dy)
    
    return lap_M