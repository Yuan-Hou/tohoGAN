import torchvision
from AvatarDataloader import *
from torch.utils.data import DataLoader
from GAN import *
from torch import nn
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用 {}".format(device))

imgPath = r'thDataset'
batchSize = 16
epochs = 10000

allData = AvatarData(imgPath)

dataloader = DataLoader(dataset=allData,batch_size=batchSize,drop_last=True)

G,D = Gen(),Disc()


G = G.to(device)
D = D.to(device)
    
if(os.path.exists("G.pth")):
    G.load_state_dict(torch.load("G.pth",map_location=torch.device(device)))
    
if(os.path.exists("D.pth")):
    D.load_state_dict(torch.load("D.pth",map_location=torch.device(device)))
    
optG = torch.optim.Adam(G.parameters(),lr=0.02)
optD = torch.optim.ASGD(D.parameters(),lr=0.002)

lossFn = nn.BCELoss().to(device)

allTrue = torch.ones(batchSize).to(device)
allFalse = torch.zeros(batchSize).to(device)

noises = torch.randn(batchSize,256,1,1).to(device)

exampleNoise = torch.randn(1,256,1,1).to(device)

for epoch in range(0,epochs):
    totalDLoss = 0
    totalGLoss = 0
    if (epoch)%50==0:
        torch.save(D.state_dict(),"D.pth")
        torch.save(G.state_dict(),"G.pth")
        G.eval()
        gen = G(exampleNoise)
        torchvision.utils.save_image(gen.data,"Gen/%s.png"%(epoch),normalize = True)
        G.train()
    D.train()
    optG.zero_grad()
        
    noises.data.copy_(torch.randn(batchSize, 256, 1, 1))
    fakeImg = G(noises)
    output = D(fakeImg)
    pretendLoss = lossFn(output,allTrue)
    pretendLoss.backward()
    optG.step()
    
    totalGLoss += pretendLoss
    if(epoch%5==0):
        for n,img in enumerate(dataloader):
            optD.zero_grad()

            realOut = D(img)

            realLoss = lossFn(realOut,allTrue)
            realLoss.backward()

            noises = noises.detach()
            fakeImg = G(noises).detach()
            fakeOut = D(fakeImg)
            fakeLoss = lossFn(fakeOut,allFalse)
            fakeLoss.backward()
            optD.step()

            totalDLoss += realLoss+fakeLoss
    print("%d:%.3f,%.3f"%(epoch,totalDLoss,totalGLoss))
    
        
        
            