import imghdr
from ntpath import join
import certifi
from bs4 import BeautifulSoup
import os
import urllib3

wikiURL = r'https://thwiki.cc/%E5%AE%98%E6%96%B9%E8%A7%92%E8%89%B2%E5%88%97%E8%A1%A8' # 官方角色列表（带Q版图片）
noImgURL = r"'https://upload.thwiki.cc/4/43/%E6%97%A0%E7%AB%8B%E7%BB%98%EF%BC%88Q%E7%89%88%E7%AB%8B%E7%BB%98%EF%BC%89.png'"
targ = r'thDataset'

http = urllib3.PoolManager(ca_certs=certifi.where())
resp = http.request('GET',wikiURL)
soup = BeautifulSoup(resp.data,'html.parser')

imgDOMs = soup.find(attrs={"class":"chara-body"}).find_all(name="img")

for imgDOM in imgDOMs:
    if(imgDOM['v-lazy']!=noImgURL):
        name = imgDOM[':key'].strip("'")
        imgURL = imgDOM['v-lazy'].strip("'")
        print(name,imgURL)
        imgResp = http.request('GET',imgURL)
        with open(os.path.join(targ,name+".png"),'wb') as f:
            f.write(imgResp.data)
        
