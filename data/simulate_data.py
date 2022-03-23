import os
import sys
import shutil

import torch
torch.manual_seed(42)

import numpy as np
np.random.seed(42)

sys.path.append('../')
from dynamics import*

# --------------- Physical prameters ---------------
L_x = 1E+6              # Length of domain in x-direction
L_y = 1E+6              # Length of domain in y-direction
g = 9.81                # Acceleration of gravity [m/s^2]
H = 100                 # Depth of fluid [m]

# --------------- Computational prameters ---------------
N_x = 64                            # Number of grid points in x-direction
N_y = 64                            # Number of grid points in y-direction
dx = L_x/(N_x - 1)                  # Grid spacing in x-direction
dy = L_y/(N_y - 1)                  # Grid spacing in y-direction

x = torch.linspace(-L_x/2, L_x/2, N_x)  # Array with x-points
y = torch.linspace(-L_y/2, L_y/2, N_y)  # Array with y-points
Xx, Yy = torch.meshgrid(x, y) 

n_trajectory = 100  #number of trajectories to save
T = 20  # temporal length of trajectory

# --------------- Directory management ---------------
directory = './ground_truth/'

if os.path.exists(directory):
    shutil.rmtree(directory)
os.makedirs(directory)

# --------------- Dynamics ---------------
dynamics=SW(dx=dx,dy=dy,H=H,g=g)

# --------------- Loop ---------------
for i in range(n_trajectory):
    
    # --------------- random initial conditions ---------------
    xx=np.random.uniform(3,7,1)*np.random.choice([-1,1], size=1, p=[0.5, 0.5])
    yy=np.random.uniform(3,7,1)*np.random.choice([-1,1], size=1, p=[0.5, 0.5])
    s=np.random.uniform(5,10,1)*1E+4
    
    traj = torch.zeros((T,3,N_x,N_y))
    X0=torch.zeros((3,N_x,N_y))
    X0[0,:,:]=4*torch.exp(-((Xx-L_x/xx)**2/(2*(s)**2) + 
                            (Yy-L_y/yy)**2/(2*(s)**2)))
    
    # --------------- 1st integration run to reach equilibrium ---------------
    for t in range(1000):
            X0=dynamics.forward(X0)
            
    # --------------- 2nd integration - final trjaectory ---------------
    traj[0,:,:,:] = X0
    
    for t in range(1,T):
        traj[t,:,:,:] = dynamics.forward(traj[t-1,:,:,:])
                
    traj = traj.numpy()
    np.save(directory+'{0:04}'.format(int(i)), traj)