import torch


class smooth_regul():
    
    def __init__(self,alpha,beta,dx,dy):

        self.alpha=alpha
        self.beta=beta
        self.dx=dx
        self.dy=dy

    def J(self,X):
        
        dx=self.dx
        dy=self.dy
        
        X0=X[0,:,:,:]

        U_0 = torch.zeros((X0.shape[1]+2,X0.shape[2]+2), dtype=torch.float64)
        V_0 = torch.zeros((X0.shape[1]+2,X0.shape[2]+2), dtype=torch.float64)

        U_0[1:(U_0.shape[0]-1),1:(U_0.shape[1]-1)] = X0[1,:,:]
        V_0[1:(V_0.shape[0]-1),1:(V_0.shape[1]-1)] = X0[2,:,:]

        # grad norm
        grad_ux,grad_uy = grad_mat(U_0,dx,dy)
        grad_vx,grad_vy = grad_mat(V_0,dx,dy)

        j_grad = 0.5*self.alpha*(
        torch.sum(grad_ux**2) +torch.sum(grad_uy**2)+
        torch.sum(grad_vx**2) +torch.sum(grad_vy**2))

        # div norm
        div_w0 = div_mat(V_0,U_0,dy,dx)

        j_div = 0.5*self.beta*torch.sum(div_w0**2)

        # j                            
        j=j_grad+j_div

        return j
    
def cut_border(M):
    
    M_moinsbord = torch.zeros((M.shape[0]-2,M.shape[1]-2))
    M_moinsbord = M[1:(M.shape[0]-1),1:(M.shape[1]-1)]
    
    return M_moinsbord



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