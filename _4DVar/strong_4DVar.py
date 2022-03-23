import torch
import torch.nn as nn
import torch.optim as optim

class strong_4DVar():
    
    
    def __init__(self, dynamics, r, regul=None,
                 optimizer=optim.LBFGS(
                     [torch.zeros(0)],
                     lr=1, 
                     max_iter=150)
                ):
                                         #tolerance_grad=0,
                                         #tolerance_change=0)
        
        self.dynamics = dynamics
        self.optimizer = optimizer
        self.r = r
        
        self.regul=regul
        self.n_iter = 0
        self.losses=[]
        self.convergence=1
    
    def Forward(self, X0):
        
        # Initialize state
        X = torch.zeros(self.DAw)
        X[0,:,:,:] = X0
        
        # Time integration
        for t in range(1,self.T):
            X[t,:,:,:] = self.dynamics.forward(X[(t-1),:,:,:].clone())
            
        return X
    
    def decimation(self, tensor):
        
        k_size=(self.r,self.r)
        _decimation= nn.AvgPool2d(k_size, stride=(self.r, self.r), padding=0)
        
        d_tensor = tensor.unsqueeze(0)
        d_tensor = _decimation(d_tensor).squeeze(0)
        
        return d_tensor
        
    def J_obs(self, X, Rm1, Obs):
        
        # decimate X
        X_d = torch.zeros(Rm1.shape)
        
        for i in range(X_d.shape[0]):
            X_d[i,0,:,:]=self.decimation(X[i,0,:,:])
            
        # Quadratic observational error
        j = ((Obs-X_d)*Rm1*(Obs-X_d)).mean()
    
        return j

    def fit(self, interp_Obs, Obs, Rm1):
        
        # Obs & Covariance
        self.DAw = interp_Obs.shape
        self.T = self.DAw[0]
        
        #self.interp_Obs = interp_Obs
        self.Obs=Obs
        self.Rm1 = Rm1
        
        # Background // first obs by default
        X_b = torch.Tensor(interp_Obs[0,:,:,:])
        
        # eps_b, control paramaters
        self.eps_b = torch.zeros(X_b.shape)
        self.eps_b.requires_grad = True
        self.optimizer.param_groups[0]['params'][0] = self.eps_b
        
        def closure():
            
            self.optimizer.zero_grad()
            X0 = X_b + self.eps_b
            X = self.Forward(X0)
            
            if self.regul == None:
                loss  = self.J_obs(X,self.Rm1,self.Obs)
            else:
                loss  = self.J_obs(X,self.Rm1,self.Obs)+self.regul.J(X)

            # check for NaN
            if torch.isnan(loss).item() != 0:          
                print('Nan loss: failed to converge')
                self.convergence=0
                loss = torch.zeros(1,requires_grad = True)
            
            loss.backward(retain_graph=True)
            
            self.n_iter = self.n_iter + 1
            self.losses.append(loss)
            
            return loss
        
        loss = self.optimizer.step(closure)
        
        # Full state
        self.X = self.Forward(X_b + self.eps_b)
        
        # Initial condition
        self.initial_condition = self.X[0,:,:,:].detach()
        
        
    def forecast(self, n_step=10, d=False):

        X_forecast = self.X[-1,:,:,:].detach()

        for i in range (0, n_step):
            X_forecast = self.dynamics.forward(X_forecast)
            
        if d:  
            X_forecast = self.decimation(X_forecast)
            
        return X_forecast