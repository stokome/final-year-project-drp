import torch
import torch.nn as nn

class Variational_Auto_Encoder(nn.Module):

    def __init__(self, device, indim, outdim=512):
        super(Variational_Auto_Encoder, self).__init__()
        self.encoder = Encoder(device=device,indim=indim,outdim=outdim)
        self.decoder = Decoder(device=device,outdim=indim,indim=outdim)
        self.mean_layer = nn.Linear(1024, outdim)
        self.logvar_layer = nn.Linear(1024, outdim)
        self.device = device

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)      
        z = mean + var*epsilon
        return z

    def forward(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar

    def output(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        z = self.reparameterization(mean, logvar)
        return z

class Encoder(nn.Module):
    def __init__(self,device, indim, outdim=512):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(indim, 4089, device=device)
        self.linear2 = nn.Linear(4089, 1024, device=device)

    def forward(self, x):
        x = nn.LeakyReLU()(self.linear1(x))
        x = nn.LeakyReLU()(self.linear2(x)) 
        return x
        
class Decoder(nn.Module):
    def __init__(self,device,outdim, indim=512):
        super(Decoder, self).__init__()
        self.linear3 = nn.Linear(indim, 1024, device=device)
        self.linear2 = nn.Linear(1024, 4089, device=device)
        self.linear1 = nn.Linear(4089, outdim, device=device)
    
    def forward(self, x):
        x = nn.LeakyReLU()(self.linear3(x))
        x = nn.LeakyReLU()(self.linear2(x))
        x = nn.Sigmoid()(self.linear1(x)) 
        return x