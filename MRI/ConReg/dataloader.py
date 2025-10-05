
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
from Augmentation import RandomCrop, CenterCrop, RandomFlip
from skimage.transform import resize





class MRIloader():
    def __init__(self, train,normalize,randomCrop,randomFlip,flipProbability,cropDim,year="1yr"):
        self.train = train
        self.normalize = normalize
        self.randomCrop = randomCrop
        self.randomFlip = randomFlip
        self.flipProbability = flipProbability
        self.cropDim = cropDim
        self.year = year

        self.data_path = "/gpfs/data/denizlab/Datasets/OAI/COR_IW_TSE/"
        
        
    def __len__(self):
        return len(self.train)
    def preprocess(self,pre_image):
        if pre_image.shape[2]<self.cropDim[2]:
            pre_image = self.padding_image(pre_image)
            
            
        # normalize
        if self.normalize:
            pre_image = self.normalize_MRIs(pre_image)
        # Augmentation
        if self.randomFlip:
            pre_image = RandomFlip(image=pre_image,p=0.5).horizontal_flip(p=self.flipProbability)
        if self.randomCrop:
            pre_image = RandomCrop(pre_image).crop_along_hieght_width_depth(self.cropDim)
        else:
            pre_image = CenterCrop(image=pre_image).crop(size = self.cropDim)
        #print(ID,pre_image.shape)
        
        pre_image = np.expand_dims(pre_image,0)

        return pre_image

    def normalize_MRIs(self,image):
        mean = np.mean(image)
        std = np.std(image)
        image -= mean
        #image -= 95.09
        image /= std
        #image /= 86.38
        return image
    def padding_image(self,data):
        l,w,h = data.shape
        images = np.zeros((l,w,self.cropDim[2]))
        zstart = int(np.ceil((self.cropDim[2]-data.shape[2])/2))

        images[:,:,zstart:zstart + h] = data
        return images
    def __getitem__(self, idx):
        locs = self.train.iloc[idx]
        id_val = locs["ID"]
        label1 = locs[self.year+'Label1']
        label0 = locs[self.year+'Label0']

        Side = locs["Side"]

        if Side ==0:
            f_name = "_LEFT_COR_IW_TSE.hdf5"
        else:
            f_name = "_RIGHT_COR_IW_TSE.hdf5"

        
        months = ["00m",locs["last_month"]]

        images=[]
        for i in range(2):
            full_path = self.data_path+months[i]+"/"+str(id_val)+"_"+months[i]+f_name
            f = h5py.File(full_path, "r")
            pre_image = f["data"][:].astype('float64')
            pre_image = self.preprocess(pre_image)
            images.append(torch.from_numpy(pre_image.copy()))
            
        seq_len = len(images)



        if label0==label1:
            siam_label = 1
        else:
            siam_label = 0


        

        return torch.stack(images,0),int(label0),int(label1),siam_label

        

class MOSTloader():
    def __init__(self, train,normalize,randomCrop,randomFlip,flipProbability,cropDim,year="1yr"):
        self.train = train
        self.normalize = normalize
        self.randomCrop = randomCrop
        self.randomFlip = randomFlip
        self.flipProbability = flipProbability
        self.cropDim = cropDim
        self.year = year

        self.data_path = "/gpfs/data/denizlab/Datasets/MOST/HR_COR_STIR/"
        
        
    def __len__(self):
        return len(self.train)
    def preprocess(self,pre_image):
        if pre_image.shape[2]<self.cropDim[2]:
            pre_image = self.padding_image(pre_image,dim=2)

        pre_image = resize(pre_image, (384, 384,self.cropDim[2]),anti_aliasing=True)


        
            
            
        # normalize
        if self.normalize:
            pre_image = self.normalize_MRIs(pre_image)
        # Augmentation
        if self.randomFlip:
            pre_image = RandomFlip(image=pre_image,p=0.5).horizontal_flip(p=self.flipProbability)
        if self.randomCrop:
            pre_image = RandomCrop(pre_image).crop_along_hieght_width_depth(self.cropDim)
        else:
            pre_image = CenterCrop(image=pre_image).crop(size = self.cropDim)
        #print(ID,pre_image.shape)
        
        pre_image = np.expand_dims(pre_image,0)

        return pre_image

    def normalize_MRIs(self,image):
        mean = np.mean(image)
        std = np.std(image)
        image -= mean
        #image -= 95.09
        image /= std
        #image /= 86.38
        return image
    def padding_image(self,data,dim):
        
        if dim==2:
            l,w,h = data.shape
            images = np.zeros((l,w,self.cropDim[2]))
            zstart = int(np.ceil((self.cropDim[2]-data.shape[2])/2))

            images[:,:,zstart:zstart + h] = data
        elif dim==1:
            l,w,h = data.shape
            images = np.zeros((l,self.cropDim[1],h))
            ystart = int(np.ceil((self.cropDim[1]-data.shape[1])/2))

            images[:,ystart:ystart + w,:] = data
        elif dim==0:
            l,w,h = data.shape
            images = np.zeros((self.cropDim[0],w,h))
            xstart = int(np.ceil((self.cropDim[0]-data.shape[0])/2))

            images[xstart:xstart + l,:,:] = data


        return images
    def __getitem__(self, idx):
        locs = self.train.iloc[idx]
        most_id = str(locs["MOST_ID"])
        actrostic = str(locs["ACROSTIC"])
        label1 = locs[self.year+'Label1']
        label0 = locs[self.year+'Label0']

        Side = locs["KNEE"]

        if Side ==0:
            f_name = "_LEFT_HR_COR_STIR.hdf5"
        else:
            f_name = "_RIGHT_HR_COR_STIR.hdf5"

        
        months = ["V0",locs["last_month"]]

        images=[]
        for i in range(2):
            full_path = self.data_path+months[i]+"/"+most_id+"_"+actrostic+"_"+months[i]+f_name
            f = h5py.File(full_path, "r")
            pre_image = f["data"][:].astype('float64')
            pre_image = self.preprocess(pre_image)
            images.append(torch.from_numpy(pre_image.copy()))
            
        seq_len = len(images)



        if label0==label1:
            siam_label = 1
        else:
            siam_label = 0


        #print(images[0].shape)

        if locs["last_month"] == "V0":
            siam_label=-1

        

        return torch.stack(images,0),int(label0),int(label1),siam_label
        

 
        
        
        
        



