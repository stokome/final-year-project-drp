import os
import torch.nn as nn

class Deep_Auto_Encoder(nn.Module):
    def __init__(self, device, indim, outdim=512):
        super(Deep_Auto_Encoder, self).__init__()
        self.encoder = Encoder(device=device,indim=indim,outdim=outdim)
        self.decoder = Decoder(device=device,outdim=indim,indim= outdim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def output(self, x):
        return self.encoder(x)


class Encoder(nn.Module):
    def __init__(self,device, indim, outdim=512):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(indim,8192,device=device)
        self.linear2 = nn.Linear(8192,2048,device=device)
        self.linear3 = nn.Linear(2048,outdim,device=device)

    def forward(self, x):
        x = nn.SELU()(self.linear1(x))
        x = nn.SELU()(self.linear2(x))
        x = nn.Sigmoid()(self.linear3(x))
        return x


class Decoder(nn.Module):
    def __init__(self,device,outdim, indim=512):
        super(Decoder, self).__init__()
        self.linear3 = nn.Linear(indim,2048,device=device)
        self.linear2 = nn.Linear(2048,8192,device=device)
        self.linear1 = nn.Linear(8192,outdim,device=device)

    def forward(self, x):
        x = nn.SELU()(self.linear3(x))
        x = nn.SELU()(self.linear2(x))
        x = nn.Sigmoid()(self.linear1(x))
        return x