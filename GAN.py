from torch import nn, sigmoid
import torch
from torchvision import models

class Gen(nn.Module):
    def __init__(self):
        super(Gen,self).__init__()
        self.main  = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=256,out_channels=1024,kernel_size=7,padding=6,bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            # 7
            nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 14
            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 28
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 56
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 112
            nn.ConvTranspose2d(in_channels=64,out_channels=4,kernel_size=4,stride=2,padding=1,bias=False),
            nn.Tanh()
        )
    def forward(self,x):
        return self.main(x)

class ConvBNLeaky(nn.Module):
    def __init__(self,inChannels,outChannels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False)
        self.BN = nn.BatchNorm2d(128)
        self.ReLU = nn.LeakyReLU(0.2,inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.ReLU(x)
        return x
    
class Disc(nn.Module):
    def __init__(self):
        super(Disc,self).__init__()
#         self.main = nn.Sequential(
#             # 224
#             nn.Conv2d(in_channels=4,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False),
#             nn.LeakyReLU(0.3,inplace=True),
#             # 112
#             nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.3,inplace=True),
#             # 56
#             nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=1,bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.3,inplace=True),
#             # 28
#             nn.Conv2d(in_channels=256,out_channels=512,kernel_size=4,stride=2,padding=1,bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.3,inplace=True),
#             # 14
#             nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=4,stride=2,padding=1,bias=False),
#             nn.BatchNorm2d(1024),
#             nn.LeakyReLU(0.3,inplace=True),
#             # 7
#             nn.Conv2d(in_channels=1024,out_channels=1,kernel_size=7,stride=1,padding=0,bias=False),
#             # 1
#             nn.Sigmoid()
            
            
#         )
#        self.fc = nn.Linear(256,1)
    def forward(self,x):
        return self.main(x).view(-1)

