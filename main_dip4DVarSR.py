import os
import torch
import numpy as np

from utils import*
from _DIP4DVarSR import*
from dynamics import*

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ',device)

torch.manual_seed(42)

if not os.path.exists('./data/results/'):
    os.makedirs('./data/results/')

# EXPERIMENT PARAMETERS

dynamics=SW(device=device)
T = 10
subsample = 3

R = [1,2,4,8] #donwscaling factor
Noise_p = [0, 0.01, 0.02, 0.03] #noise percentage


dataset_ex = Obs_set(r=1, T=T, subsample=subsample, sigma = 0, 
                 root_dir='./data/ground_truth/')

results = np.zeros((len(dataset_ex), 2, len(R), len(Noise_p), 11))

n_epoch = 1000
lr = 0.001
n_z=100

mu = torch.randn(n_z, 1, 1, device=device)
sigma = torch.randn(n_z, 1, 1, device=device)

#best_params_dip = np.load('./data/hyperparams/best_params_dip.npy',allow_pickle='TRUE').item()

for r in R:

    #lr = best_params_dip['r='+str(r)]['lr']
    #n_epoch = best_params_dip['r='+str(r)]['n_epoch']
    #n_z = best_params_dip['r='+str(r)]['n_z']
    
    for noise_p in Noise_p:
        print('########## r: '+str(r)+' // noise percentage :'+str(noise_p*100))
        
        dataset = Obs_set(r=r, T=T, subsample=subsample, sigma = noise_p, 
                 root_dir='./data/ground_truth/')
        
        for i in range(len(dataset)):
        
            Obs, _, Rm1, traj = dataset.__getitem__(i)
            
            Obs=Obs.to(device)
            Rm1=Rm1.to(device)
            #traj.to(device)
            
            eta0_truth = traj[0,0,:,:]
            w0_truth = traj[0,1:,:,:]
            
            netG=Generator(nz=n_z)
            netG.apply(weights_init)
            netG.to(device)
            
            optimizer = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
        
            # DIP
            assim = dip_4DVarSR(r=r, net=netG, dynamics=dynamics, 
                    optimizer=optimizer, device=device, n_epoch=n_epoch)
            
            assim.fit(mu=mu, sigma=sigma, Obs=Obs, Rm1=Rm1)
            
                # Assim results
            eta0_deep=assim.initial_condition[0,:,:]
            w0_deep=assim.initial_condition[1:,:,:]
                
            results[i,0, R.index(r), Noise_p.index(noise_p),0]=RMSE(
                eta0_truth, eta0_deep.cpu())
            results[i,0, R.index(r), Noise_p.index(noise_p),1]=SSIM(
                eta0_truth, eta0_deep.cpu())
            results[i,0, R.index(r), Noise_p.index(noise_p),2]=EPE(
                w0_truth, w0_deep.cpu())
            results[i,0, R.index(r), Noise_p.index(noise_p),3]=AAE(
                w0_truth, w0_deep.cpu())
            
                # Forecast results
            eta_forecast_t1=assim.forecast(1)[0,:,:]
            eta_forecast_t5=assim.forecast(5)[0,:,:]
            
            results[i,0, R.index(r), Noise_p.index(noise_p),4]=RMSE(
                traj[T,0,:,:], eta_forecast_t1.cpu())
            results[i,0, R.index(r), Noise_p.index(noise_p),5]=SSIM(
                traj[T,0,:,:], eta_forecast_t1.cpu())
            results[i,0, R.index(r), Noise_p.index(noise_p),6]=RMSE(
                traj[T+4,0,:,:], eta_forecast_t5.cpu())
            results[i,0, R.index(r), Noise_p.index(noise_p),7]=SSIM(
                traj[T+4,0,:,:], eta_forecast_t5.cpu())
            
                # Assim w stats
            results[i,0, R.index(r), Noise_p.index(noise_p),8]=norm_gradw(w0_deep.cpu())
            results[i,0, R.index(r), Noise_p.index(noise_p),9]=norm_divw(w0_deep.cpu())
            results[i,0, R.index(r), Noise_p.index(noise_p),10]=norm_lapw(w0_deep.cpu())
            
            np.save('./data/results/main_dip4DVarSR.npy',results)