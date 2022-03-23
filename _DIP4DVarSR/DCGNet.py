import torch.nn as nn

# custom weights initialization called on netG
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

        
class Generator(nn.Module):
    def __init__(self, nz, nc=3,ngf=64, tanh=True, batchnorm=True):
        super(Generator, self).__init__()
        
        # Number of channels in the training images. (3 for eta,u,v)
        self.nc=nc
        # Size of feature maps in generator
        self.ngf=ngf
        # Size of z latent vector (i.e. size of generator input)
        self.nz=nz
        # batchnorm
        self.batchnorm=batchnorm
        # tanh activation
        self.tanh=tanh
        
        
        self.main=[]
        self.main.append(nn.ConvTranspose2d( self.nz, self.ngf * 8, 4, 1, 0, bias=False) )
        if self.batchnorm:
            self.main.append( nn.BatchNorm2d(self.ngf * 8) )
        self.main.append( nn.ReLU(True) )
        
        # state size. (ngf*8) x 4 x 4
        
        self.main.append( nn.Upsample(scale_factor = 2, mode='bilinear',align_corners=False) )
        self.main.append( nn.ReflectionPad2d(1) )
        self.main.append( nn.Conv2d(self.ngf * 8, self.ngf * 4,kernel_size=3, stride=1, padding=0) )
        if self.batchnorm:
            self.main.append( nn.BatchNorm2d(self.ngf * 4) )
        self.main.append( nn.ReLU(True) )
        
        # state size. (ngf*4) x 8 x 8
        
        self.main.append( nn.Upsample(scale_factor = 2, mode='bilinear',align_corners=False) )
        self.main.append( nn.ReflectionPad2d(1) )
        self.main.append( nn.Conv2d(self.ngf * 4, self.ngf * 2,kernel_size=3, stride=1, padding=0) )
        if self.batchnorm:
            self.main.append( nn.BatchNorm2d(self.ngf * 2) )
        self.main.append( nn.ReLU(True) )
        
        # state size. (ngf*2) x 16 x 16
        
        self.main.append( nn.Upsample(scale_factor = 2, mode='bilinear',align_corners=False) )
        self.main.append( nn.ReflectionPad2d(1) )
        self.main.append( nn.Conv2d(self.ngf * 2, self.ngf * 1,kernel_size=3, stride=1, padding=0) )
        if self.batchnorm:
            self.main.append( nn.BatchNorm2d(self.ngf) )
        self.main.append( nn.ReLU(True) )
        
        # state size. (ngf) x 32 x 32
        
        self.main.append( nn.Upsample(scale_factor = 2, mode='bilinear',align_corners=False) )
        self.main.append( nn.ReflectionPad2d(1) )
        self.main.append( nn.Conv2d(self.ngf, self.nc ,kernel_size=3, stride=1, padding=0) )
        if self.tanh:
            self.main.append( nn.Tanh() )
        
        # state size. (nc) x 64 x 64
            
        self.main =  nn.Sequential(*self.main)           

    def forward(self, input):
        return self.main(input)