'''
亲爱的晏凡同学：
你好呀，还记得我吗？我是之前给你写信的森哥。上次写信之后，一个学期已经过去了，你在那边初中的学习生活怎么样了？我的建议有用吗？不论如何，恭喜你已经成功完成了初中第一个学期的学习，这个学期也已经过了一大半了，我相信你应该比半年前更加适应初中的生活了。如果感觉在初中的生活还是有什么困难的话，记得在回信里给我说说呀。

我先说说我这学期都有什么新鲜事吧。这学期真的不是一般的忙，写完一门作业紧接着又是另一门，还有我自己报名的各种竞赛等等。但是忙中带着充实，比如说昨天我得知有一个写程序的竞赛我在前30%，得二等奖，可以得1000块钱。不过我觉得这学期做的最有意义的一件事是，我到一个老师的实验室实习，做了一个能让电脑自动看X光片，判断有没有骨折的软件，这个国内很少有人做，但是我能参与其中，感觉很荣幸，也很有意思。
那个判断骨折的软件做好之后，我们还专门去医院让医生试用了，印象很深的是，试用完软件之后医生还夸我做的很了不起，给老师说必须得让我读上博士，虽然是客套话，但是我听了还是自豪了好长时间，感觉自己这么忙也值了。

小晏虽然现在才七年级，也一定要好好学习，成绩好坏、偏科与否关系不大，重要的还得是态度。只要足够努力，除了老天能看到，你的老师、父母也能感受到，他们多少也都会向努力的你给予帮助的。如果你在学习方法上有什么想问的，在回信里随便问我就行，我一定会认真回答的。

祝
学业有成，天天开心！

森哥
2022年4月29日
'''

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
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5,0.5), (0.5, 0.5, 0.5,0.5))
            ]
        )
        extendTrans = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.transforms.RandomHorizontalFlip(1),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5,0.5), (0.5, 0.5, 0.5,0.5))
            ]
        )
        for p,_,fs in os.walk(imgPath):
            for f in fs:
                fp = str(os.path.join(p,f))
                print(fp)
                img = cv.imdecode(np.fromfile(fp,dtype=np.uint8),-1)
                img = cv.cvtColor(img,cv.COLOR_RGBA2BGRA)
                
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

