import os
import torch
import pandas as pd
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import copy
import cv2
from pathlib import Path
from torch.nn import functional as F
import torch.nn as nn


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)   
    img = img / 255
    return torch.from_numpy(img).float()
def show_fig(dataset,i):
    if dataset.mode == "train":
        a,b = dataset[i]
        a = a.view(101,101)
        b = b.view(101,101)
        print(a.shape)
        fig,(ax1,ax2) = plt.subplots(1, 2, figsize=(15,5))
        ax1.imshow(a,cmap='Greys',  interpolation='nearest')
        ax2.imshow(b,cmap='Greys',  interpolation='nearest')
    else:
        a = dataset[i]
        print(a.shape)
        fig,ax1 = plt.subplots(1, 1, figsize=(15,5))
        ax1.imshow(a,cmap='Greys',  interpolation='nearest')



# from disscusion at https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63984#382657
def get_mask_type(mask):
    border = 10
    outer = np.zeros((101-2*border, 101-2*border), np.float32)
    outer = cv2.copyMakeBorder(outer, border, border, border, border, borderType = cv2.BORDER_CONSTANT, value = 1)
    outer = torch.from_numpy(outer)
    cover = (mask>0.5).sum()
    if cover < 8:
        return 0 # empty
    if cover == ((mask*outer) > 0.5).sum():
        return 1 #border
    if not (mask - mask[0]).byte().any():
        return 2 #vertical

    percentage = cover.float()/(101*101)
    if percentage < 0.15:
        return 3
    elif percentage < 0.25:
        return 4
    elif percentage < 0.50:
        return 5
    elif percentage < 0.75:
        return 6
    else:
        return 7


class SaltIdentification(Dataset):

    def __init__(self, mode = "train", path = None, transform=None, name = None, datalist = None, preload = False):
        # mode: train or test
        # name: which fold
        # datalist: which data to use
        # perload: the complete dataset
        Dataset.__init__(self)
        self.path = path
        self.mode = mode
        self.transform = transform
        if name:
            self.name = name
        else:
            self.name = self.mode
        self.image_folder = os.path.join(self.path, "images")
        self.mask_folder = os.path.join(self.path, "masks")
        if self.mode == "train":
            if preload is False:
                self.data = pd.read_csv(os.path.join(self.path,"train.csv"), usecols=[1,2])
            else:
                self.data = preload.data.reindex(datalist)
                self.data = self.data.reset_index()
        else:
            self.data = pd.read_csv(os.path.join(self.path,"sample_submission.csv"), usecols=[0])
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):       
        file_id = self.data.id[index]
        image_path = os.path.join(self.image_folder, file_id + ".png")
        mask_path = os.path.join(self.mask_folder, file_id + ".png")
        
        image = load_image(image_path)
        img_size = image.size()[-2:]
        if self.mode == "train":
            mask = load_image(mask_path)
            return image.view(-1,*img_size), mask.view(-1,*img_size)
        else:
            return image.view(-1,*img_size)


# In[5]:


# saved in the csv
# data.train_df["coverage_class"] = [0] * len(data.train_df)
# for i in tqdm(range(len(data.train_df))):
#     data.train_df.coverage_class[i] = get_mask_type(data[i][1])

