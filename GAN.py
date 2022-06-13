from torch import nn, sigmoid
import torch
from torchvision import models
class CtUpsample(nn.Module):
    def __init__(self,inChannels,outChannels) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=inChannels,out_channels=outChannels,kernel_size=4,stride=2,padding=1,bias=False)
        self.BN = nn.BatchNorm2d(outChannels)
        self.ReLU = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.ReLU(x)
        return x
class Gen(nn.Module):
    def __init__(self):
        super(Gen,self).__init__()
        self.L1 = nn.Conv2d(in_channels=256,out_channels=1024,kernel_size=2,padding=2,bias=True)
        self.L2 = CtUpsample(1024,512)
        self.L3 = CtUpsample(512,256)
        self.L4 = CtUpsample(256,128)
        self.L5 = CtUpsample(128,64)
        self.L6 = nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=4,stride=2,padding=1,bias=False)
        self.tan = nn.Tanh()
        
    def forward(self,x):
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        x = self.L5(x)
        x = self.L6(x)
        x = self.tan(x)
        return x

class ConvBNLeaky(nn.Module):
    def __init__(self,inChannels,outChannels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inChannels,out_channels=outChannels,kernel_size=4,stride=2,padding=1,bias=False)
        self.BN = nn.BatchNorm2d(outChannels)
        self.ReLU = nn.LeakyReLU(0.2,inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.ReLU(x)
        return x
    
class Disc(nn.Module):
    def __init__(self):
        super(Disc,self).__init__()
        self.L1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False)
        self.ReLU = nn.LeakyReLU(0.2,inplace=True)
        self.L2 = ConvBNLeaky(64,128)
        self.L3 = ConvBNLeaky(128,256)
        self.L4 = ConvBNLeaky(256,512)
        self.L5 = ConvBNLeaky(512,1024)
        self.fc = nn.Conv2d(in_channels=1024,out_channels=1,kernel_size=2,stride=1,padding=0,bias=False)
        self.sig = nn.Sigmoid()
    def forward(self,x):
        x = self.L1(x)
        x = self.ReLU(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        x = self.L5(x)
        x = self.fc(x)
        x = self.sig(x)
        return x.view(-1)

