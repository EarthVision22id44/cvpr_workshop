import torch.nn as nn
import torch
import torch.optim as optim

class dip_4DVarSR():
    
    def __init__(self, r, net, dynamics, optimizer, device, n_epoch):
        
        self.r = r
        self.net=net
        self.dynamics = dynamics
        
        self.n_epoch=n_epoch
        self.optimizer = optimizer
        self.device = device
        
        self.losses=[]
        self.initial_condition=0
        
    def decimation(self, tensor):
        
        k_size=(self.r,self.r)
        _decimation= nn.AvgPool2d(k_size, stride=(self.r, self.r), padding=0)
        
        d_tensor = tensor.unsqueeze(0)
        d_tensor = _decimation(d_tensor).squeeze(0)
        
        return d_tensor
        
    def J_obs(self, X, Rm1, Obs):
        
        # decimate X
        X_d = torch.zeros(Rm1.shape, device=self.device)
        
        for i in range(X_d.shape[0]):
            X_d[i,0,:,:]=self.decimation(X[i,0,:,:])
            
        # Quadratic observational error
        j = ((Obs-X_d)*Rm1*(Obs-X_d)).mean()
    
        return j

    def fit(self, mu, sigma, Obs, Rm1):
        
        self.mu = mu
        self.sigma = sigma
        
        for i in range(self.n_epoch):
            
            self.net.zero_grad()
            loss=0
            z=mu+torch.rand_like(sigma)*sigma
            ic_gen=self.net((z).unsqueeze(0)).squeeze(0)
        
            X=torch.zeros((Obs.shape[0],3,64,64), device=self.device)
            X[0,:,:,:]=ic_gen
        
            for t in range(Obs.shape[0]-1):
            
                X[t+1,:,:,:]=self.dynamics.forward(X[t,:,:,:].clone())
            
            loss  = self.J_obs(X, Rm1, Obs)
            self.losses.append(loss.item())
            loss.backward(retain_graph=True)
            self.optimizer.step()
        
        # initial condition
        self.initial_condition = self.net(mu.unsqueeze(0)).squeeze(0).detach()
        
        # full trajectory
        self.X = torch.zeros((Obs.shape[0],3,64,64), device=self.device)
        self.X[0,:,:,:] = self.initial_condition
        
        for t in range(Obs.shape[0]-1):
                self.X[t+1,:,:,:]=self.dynamics.forward(self.X[t,:,:,:].clone())
        
    def forecast(self, n_step=10, d=False):

        X_forecast = self.X[-1,:,:,:].detach().to(self.device)

        for i in range (0, n_step):
            X_forecast = self.dynamics.forward(X_forecast)
            
        if d:  
            X_forecast = self.decimation(X_forecast)
            
        return X_forecast
    
    def proba_forecast(self, n_step=10, n_sample=50):
        
        shape = (n_sample, self.X.shape[1], self.X.shape[2], self.X.shape[3])
        X_forecast = torch.zeros(shape, device=self.device)
        
        for n in range(n_sample):
            z=self.mu+torch.rand_like(self.sigma)*self.sigma
            X_forecast[n,:,:,:] = self.net((z).unsqueeze(0)).squeeze(0)
            
            for i in range (0, n_step):
                X_forecast[n,:,:,:] = self.dynamics.forward(X_forecast[n,:,:,:])
            
        return X_forecast