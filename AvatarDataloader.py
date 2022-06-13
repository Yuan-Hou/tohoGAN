from random import shuffle
import cv2 as cv
import torch
from torchvision import transforms
import numpy as np
import os
from torch.utils.data import Dataset

tensor2img = transforms.ToPILImage()

class AvatarData(Dataset):
    def __init__(self,imgPath):
        super(AvatarData,self).__init__()
        self.dataset=[]
        normTrans = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64,64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        extendTrans = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64,64)),
                transforms.transforms.RandomHorizontalFlip(1),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        for p,_,fs in os.walk(imgPath):
            for f in fs:
                fp = str(os.path.join(p,f))
                print(fp)
                img = cv.imdecode(np.fromfile(fp,dtype=np.uint8),-1)
                img = cv.cvtColor(img,cv.COLOR_RGBA2BGR)
                
                img1 = normTrans(img).float()
                img2 = extendTrans(img).float()
                if(torch.cuda.is_available()):
                    img1 = img1.cuda()
                    img2 = img2.cuda()
                self.dataset.append(img1)
                self.dataset.append(img2)
        shuffle(self.dataset)
                
        
    def __getitem__(self,idx):
        return self.dataset[idx]
    def __len__(self):
        return len(self.dataset)

