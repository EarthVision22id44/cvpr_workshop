import os
import torch
import numpy as np

from utils import*
from _4DVar import*
from dynamics import*

if not os.path.exists('./data/results/'):
    os.makedirs('./data/results/')

# EXPERIMENT PARAMETERS

dynamics=SW()
T = 10
subsample = 3

R = [1,2,4,8] #donwscaling factor
Noise_p = [0, 0.01, 0.02, 0.03] #noise percentage

best_params_4dv = np.load('./data/hyperparams/best_params_4dv.npy',allow_pickle='TRUE').item()
#print(read_dictionary)


# RESULTS
# 2 4Dvar regul/no-regul
# 8 (4+4)  4 Assim score metrics (SSIM, RMSE, EPE, AAE) // 2x2 forecast t+1/5 (RMSE/SSIM)

dataset_ex = Obs_set(r=1, T=10, subsample=3, sigma = 0, 
                 root_dir='./data/ground_truth/')

results = np.zeros((len(dataset_ex), 2, len(R), len(Noise_p), 11))

for r in R:
    
    alpha = best_params_4dv['r='+str(r)]['alpha']
    beta = best_params_4dv['r='+str(r)]['beta']
    
    for noise_p in Noise_p:
        print('########## r: '+str(r)+' // noise percentage :'+str(noise_p*100))
        
        dataset = Obs_set(r=r, T=T, subsample=subsample, sigma = noise_p, 
                 root_dir='./data/ground_truth/')
        
        for i in range(len(dataset)):
        
            Obs, interp_Obs, Rm1, traj = dataset.__getitem__(i)
            
            eta0_truth = traj[0,0,:,:]
            w0_truth = traj[0,1:,:,:]
        
            # strong 4DVar
            assim = strong_4DVar(r=r, dynamics=dynamics)
                     #optimizer=optim.LBFGS([torch.zeros(0)],lr=1, max_iter=1))
            assim.fit(interp_Obs=interp_Obs, Obs=Obs, Rm1=Rm1)
            
                # Assim results
            eta0_4dvar=assim.initial_condition[0,:,:]
            w0_4dvar=assim.initial_condition[1:,:,:]
                
            results[i,0, R.index(r), Noise_p.index(noise_p),0]=RMSE(eta0_truth, eta0_4dvar)
            results[i,0, R.index(r), Noise_p.index(noise_p),1]=SSIM(eta0_truth, eta0_4dvar)
            results[i,0, R.index(r), Noise_p.index(noise_p),2]=EPE(w0_truth, w0_4dvar)
            results[i,0, R.index(r), Noise_p.index(noise_p),3]=AAE(w0_truth, w0_4dvar)
            
                # Forecast results
            eta_forecast_t1=assim.forecast(1)[0,:,:]
            eta_forecast_t5=assim.forecast(5)[0,:,:]
            
            results[i,0, R.index(r), Noise_p.index(noise_p),4]=RMSE(traj[T,0,:,:], eta_forecast_t1)
            results[i,0, R.index(r), Noise_p.index(noise_p),5]=SSIM(traj[T,0,:,:], eta_forecast_t1)
            results[i,0, R.index(r), Noise_p.index(noise_p),6]=RMSE(traj[T+4,0,:,:], eta_forecast_t5)
            results[i,0, R.index(r), Noise_p.index(noise_p),7]=SSIM(traj[T+4,0,:,:], eta_forecast_t5)
            
                # Assim w stats
            results[i,0, R.index(r), Noise_p.index(noise_p),8]=norm_gradw(w0_4dvar)
            results[i,0, R.index(r), Noise_p.index(noise_p),9]=norm_divw(w0_4dvar)
            results[i,0, R.index(r), Noise_p.index(noise_p),10]=norm_lapw(w0_4dvar)
        
            # strong 4DVar+regul
            smoothreg= smooth_regul(alpha=alpha, beta=beta,
                        dx=dynamics.dx,dy=dynamics.dy)
            
            assim = strong_4DVar(r=r, dynamics=dynamics, regul=smoothreg)
                     #optimizer=optim.LBFGS([torch.zeros(0)],lr=1, max_iter=1))
            assim.fit(interp_Obs=interp_Obs, Obs=Obs, Rm1=Rm1)
            
                # Assim results
            eta0_4dvar=assim.initial_condition[0,:,:]
            w0_4dvar=assim.initial_condition[1:,:,:]
            
            results[i,1, R.index(r), Noise_p.index(noise_p),0]=RMSE(eta0_truth, eta0_4dvar)
            results[i,1, R.index(r), Noise_p.index(noise_p),1]=SSIM(eta0_truth, eta0_4dvar)
            results[i,1, R.index(r), Noise_p.index(noise_p),2]=EPE(w0_truth, w0_4dvar)
            results[i,1, R.index(r), Noise_p.index(noise_p),3]=AAE(w0_truth, w0_4dvar)
            
                # Forecast results
            eta_forecast_t1=assim.forecast(1)[0,:,:]
            eta_forecast_t5=assim.forecast(5)[0,:,:]
            
            results[i,1, R.index(r), Noise_p.index(noise_p),4]=RMSE(traj[T,0,:,:], eta_forecast_t1)
            results[i,1, R.index(r), Noise_p.index(noise_p),5]=SSIM(traj[T,0,:,:], eta_forecast_t1)
            results[i,1, R.index(r), Noise_p.index(noise_p),6]=RMSE(traj[T+4,0,:,:], eta_forecast_t5)
            results[i,1, R.index(r), Noise_p.index(noise_p),7]=SSIM(traj[T+4,0,:,:], eta_forecast_t5)
            
                # Assim w stats
            results[i,1, R.index(r), Noise_p.index(noise_p),8]=norm_gradw(w0_4dvar)
            results[i,1, R.index(r), Noise_p.index(noise_p),9]=norm_divw(w0_4dvar)
            results[i,1, R.index(r), Noise_p.index(noise_p),10]=norm_lapw(w0_4dvar)
            
            np.save('./data/results/main_4DVar.npy',results)