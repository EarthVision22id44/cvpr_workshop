import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .bicubic_interp import bicubic_interp
# seed set before sampling noise - lign 66

class Obs_set(Dataset):
    
    def __init__(self, r, 
                 T=10, subsample=3, sigma = 0.01, 
                 root_dir='./data/ground_truth/'):
        
        self.r = r
        self.T = T
        self.subsample = subsample
        self.sigma = sigma
        self.root_dir=root_dir
        
        # dim - hardcoded for now
        self.Nx = 64
        self.Ny = 64
        
        # variance matrix
        self.Rm1 = torch.zeros((self.T, 3, int(self.Nx/self.r), int(self.Ny/self.r)))
        for t in range(self.T):
            if (t%self.subsample==0):
                self.Rm1[t,0,:,:] = self.Rm1[t,0,:,:]+1
        
    def __len__(self):
        return int(len(os.listdir(self.root_dir)))
    
    def H(self, traj):
        
        k_size=(self.r,self.r)
        decimation= nn.AvgPool2d(k_size, stride=(self.r, self.r), padding=0)
        
        #projection implicitly done starting from null observation
        observation=torch.zeros(self.T, 3, int(self.Nx/self.r), int(self.Ny/self.r))
        
        #decimation
        for t in range(self.T):
            if (t%self.subsample==0):
                d_traj = traj[t,0,:,:].unsqueeze(0)
                observation[t,0,:,:]=decimation(d_traj).squeeze(0)
        
        return observation
    
    def interp_obs(self, Obs):
        
        interp_observation = torch.zeros(self.T, 3, self.Nx, self.Ny)
        
        #bicubic interpolation
        for t in range(self.T):
            if (t%self.subsample==0):
                interp_observation[t,0,:,:]=bicubic_interp(Obs[t,0,:,:],
                                                          scale_factor=self.r)
        
        return interp_observation
    
    def __getitem__(self, idx):
        
        torch.manual_seed(idx)
        
        #load true trajectory
        traj = np.load(self.root_dir+'{0:04}'.format(idx)+'.npy')
        traj = torch.Tensor(traj)
        
        #apply observation operator                                                 
        Obs = self.H(traj)
                
        # add noise (only on ssh)
        if self.sigma != 0:
            obs_cc = Obs[:,0,:,:].max()-Obs[:,0,:,:].min()
            noise=torch.normal(0, self.sigma*obs_cc, Obs[:,0,:,:].shape)
            noise = noise*self.Rm1[:,0,:,:]
        else:
            noise=torch.zeros(Obs[:,0,:,:].shape)
            
        Obs[:,0,:,:] = Obs[:,0,:,:] + noise

        # bicubic interpolation for 4D-Var
        interp_Obs = self.interp_obs(Obs)
         
        return Obs, interp_Obs, self.Rm1, traj