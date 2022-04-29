from torch import nn, sigmoid
from torchvision import models

class Gen(nn.Module):
    def __init__(self):
        super(Gen,self).__init__()
        self.main  = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=7,padding=6,bias=True),
            # 7
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 14
            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 28
            nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # 56
            nn.ConvTranspose2d(in_channels=16,out_channels=8,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            # 112
            nn.ConvTranspose2d(in_channels=8,out_channels=4,kernel_size=3,stride=2,padding=1,bias=True),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.main(x)
        
class Disc(nn.Module):
    def __init__(self):
        super(Disc,self).__init__()
        self.main = models.resnet18(True)
        self.main.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.main.fc = self.fc = nn.Linear(512, 1)
    def forward(self,x):
        return nn.Sigmoid()(self.main(x)).view(-1)