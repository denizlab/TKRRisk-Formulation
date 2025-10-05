
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import h5py
import numpy as np
import XrayDataLoader
from torch.utils.data import DataLoader, Dataset


def image_loader(img_name,image_size,cropped_size,mode='val'):
    f = h5py.File(img_name, 'r')
    image = f["data"][:].astype('float64')
    image = image[..., np.newaxis]
    f.close()

    transRGB = XrayDataLoader.ToRGB()
    transResize = XrayDataLoader.Identity() if image_size == 1024 else XrayDataLoader.Resize(image_size)

    data_transforms = {'train': transforms.Compose([
                                transResize,
                                XrayDataLoader.RandomCrop(cropped_size),
                                transRGB,
                                XrayDataLoader.Normalize(),
                                XrayDataLoader.ToTensor(),
                                XrayDataLoader.RandomHorizontalFlip(),
                            ]),
                            'val': transforms.Compose([
                                transResize,
                                XrayDataLoader.CenterCrop(cropped_size),
                                transRGB,
                                XrayDataLoader.Normalize(),
                                XrayDataLoader.ToTensor(),
                            ]),
                            }
    image = data_transforms[mode](image).float()
    image = image.unsqueeze(0)
    return image




class Radiographloader():
    def __init__(self, train,image_size,cropped_size,last_index = 2,mode="train",types = "baseline",setting=1,year="1yr"):
        self.train = train
        self.n_times=last_index
        #self.all_months = ["00m","12m","24m","36m","48m"]
        self.mode = mode
        self.cropped_size = cropped_size
        self.image_size = image_size
        self.type = types
        self.setting = setting
        self.year = year

        self.data_path = "/gpfs/data/denizlab/Users/hrr288/data/"

        
    def __len__(self):
        return len(self.train)
    def __getitem__(self, idx):
        locs = self.train.iloc[idx]
        
        id_val = locs['ID']
        label1 = locs[self.year+'Label1']
        label0 = locs[self.year+'Label0']

        Side = locs["Side"]

        if Side ==0:
        	f_name = "_LEFT_KNEE.hdf5"
        else:
        	f_name = "_RIGHT_KNEE.hdf5"

        
        months = ["00m",locs["last_month"]]
       
        
        images=[]
        for i in range(self.n_times):
            full_path = self.data_path+months[i]+"/"+str(id_val)+"_"+months[i]+f_name
            image_inp = image_loader(full_path,self.image_size,self.cropped_size,mode=self.mode)
            images.append(image_inp)
            
        seq_len = len(images)
        
    
 
        #print(seq_len)
        """
        if len(images)<self.n_times:
            print("Tha fook")
            for j in range(len(images),self.n_times):
                images.append(torch.zeros_like(image_inp))
        """
        
        if label0==label1:
            siam_label = 1
        else:
            siam_label = 0

        

        return torch.squeeze(torch.stack(images,1)),int(label0),int(label1),siam_label

class MOSTloader():
    def __init__(self, train,image_size,cropped_size,last_index = 2,mode="train",types = "baseline",setting=1,year="1yr"):
        self.train = train
        self.n_times=last_index
        #self.all_months = ["00m","12m","24m","36m","48m"]
        self.mode = mode
        self.cropped_size = cropped_size
        self.image_size = image_size
        self.year = year

        self.data_path = "/gpfs/data/denizlab/Datasets/MOST/Radiographs/annotation_hg_2021-10-09_15-08-22/data/"

        
    def __len__(self):
        return len(self.train)
    def __getitem__(self, idx):
        locs = self.train.iloc[idx]
        
        id_val = locs['MOST_ID']
        acrostic = locs["ACROSTIC"]
        label0= locs[self.year+'Label0']
        label1 = locs[self.year+'Label1']
        Side = locs["KNEE"]

        if Side ==0:
            f_name = "_LEFT_KNEE.hdf5"
        else:
            f_name = "_RIGHT_KNEE.hdf5"

        
        #print(mod_df)
        months = ["V0",locs["last_month"]]
        
        images=[]
        for i in range(self.n_times):
            full_path = self.data_path+months[i]+"/"+str(id_val)+"_"+acrostic+"_"+months[i]+f_name
            image_inp = image_loader(full_path,self.image_size,self.cropped_size,mode=self.mode)
            images.append(image_inp)
            
        seq_len = len(images)

        if label0==label1:
            siam_label = 1
        else:
            siam_label = 0
        
    
 
        #print(seq_len)
        """
        if len(images)<self.n_times:
            print("Tha fook")
            for j in range(len(images),self.n_times):
                images.append(torch.zeros_like(image_inp))
        """

        return torch.squeeze(torch.stack(images,1)),label0,int(label1),siam_label


       
