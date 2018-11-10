# ## library

from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import copy
import cv2
from pathlib import Path
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import time
from torch.autograd import Variable
import torch.nn as nn
# 
from model1 import UNet
from loaddata import *
from loss import *
torch.backends.cudnn.benchmark = True 


# ## train/valid split


# local dir
path = "D:/kaggle/data"
save_path = path


saltdata = SaltIdentification(mode = "train",path = path)


# split the dataset according to coverage_class
cv_total = 4
train_all = []
evaluate_all = []
skf = StratifiedKFold(n_splits=cv_total,shuffle=True, random_state=2018)
for train_index, evaluate_index in skf.split(saltdata.data.index.values, saltdata.data.coverage_class):
    train_all.append(train_index)
    evaluate_all.append(evaluate_index)
    print(train_index.shape,evaluate_index.shape)


# ## train



def train(fold = i,mode = "pretrain",learning_rate = 1e-2, batch_size = 16, start_filters = 16, dropout_ratio = 0.5, num_epochs = 20, patience = 10):
    t0 = time.time()
    save_model_path =  os.path.join(save_path, 'model{}-1.pth'.format(i))
    # load data
    train = SaltIdentification(mode = "train",path = path,datalist = train_all[i],name = "fold1", preload = saltdata)
    val = SaltIdentification(mode = "train",path = path,datalist = evaluate_all[i],name = "fold1", preload = saltdata)
    train_loader = torch.utils.data.DataLoader(dataset=train, 
                                           batch_size=batch_size, 
                                           shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val, 
                                           batch_size=batch_size, 
                                           shuffle=True)
    # pretrain with binary entropy
    # train with symmetric_lovasz
    model = UNet(start_filters,dropout_ratio)
    if mode == "pretrain":
        model = UNet(start_filters,dropout_ratio)
        model.cuda()
        criterion = nn.BCEWithLogitsLoss()
    else:
        checkpoint = torch.load(save_model_path)
        model.load_state_dict(checkpoint)
        model.cuda()
        model.train()
        criterion = symmetric_lovasz
        save_model_path =  os.path.join(save_path, 'model{}-2.pth'.format(i))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode = "max", factor = 0.5, patience = 5,verbose=True)
    
    
    # history
    mean_train_losses = []
    mean_val_losses = []
    mean_val_accuracy = []
    lr_history = []
    best_val_accuracy = 0
    print("successfully load model in {} seconds".format(time.time() - t0))
    
    # train
    for epoch in range(num_epochs):
        model.train()
        t0 = time.time()
        train_losses = []
        val_losses = []
        val_accuracy = []
        for images, masks in train_loader: 
            images = images.cuda()
            masks = masks.cuda()
            optimizer.zero_grad()
            outputs = model(images)        
            loss = criterion(outputs, masks)
            train_losses.append(loss.data)
            loss.backward()
            optimizer.step()
        
        model.eval()
        for images, masks in val_loader:
            images = images.cuda()
            masks = masks.cuda()
            outputs = model(images)
            loss = criterion(outputs, masks)
            accuracy = my_iou_metric(outputs, masks)
            val_losses.append(loss.detach())
            val_accuracy.append(accuracy)
        
        # save histroy 
        train_losses = np.mean(train_losses)
        val_losses = np.mean(val_losses)
        val_accuracy = np.mean(val_accuracy)
        scheduler.step(val_accuracy)
        mean_train_losses.append(train_losses)
        mean_val_losses.append(val_losses)
        mean_val_accuracy.append(val_accuracy)
        
        # save the best model, early stopping
        if mean_val_accuracy == [] or val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            num_plateau = 0 
            torch.save(model.state_dict(), save_model_path)
            print("save model to {}".format(save_model_path))
        else:
            num_plateau += 1
            if num_plateau >= patience:
                print("early stopping: the val accuracy didn't improve for {} epochs".format(patience))
                break
        
        # print history
        print('Epoch: {}.time:{} seconds. Train Loss: {}. Val Loss: {}. Val Accuracy: {}'.format(epoch+1,time.time() - t0, train_losses, val_losses, val_accuracy))
    return mean_train_losses, mean_val_losses, mean_val_accuracy



for i in range(0,4):
    # hyper para
    image_file1 = 'image{}-1.png'.format(i)
    image_file2 = 'image{}-2.png'.format(i)
    learning_rate = 1e-2
    batch_size = 16
    start_filters = 16
    dropout_ratio = 0
    num_epochs = 20
    patience = 10
    # model 1
    a,b,c= train(i,learning_rate = learning_rate, batch_size = batch_size, start_filters = start_filters, dropout_ratio = dropout_ratio, 
                 num_epochs = num_epochs, patience = patience)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))
    ax_loss.plot(range(0,num_epochs), a, label="Train loss")
    ax_loss.plot(range(0,num_epochs), b, label="Validation loss")
    ax_loss.legend()
    ax_acc.plot(range(0,num_epochs), c, label="Train accuracy")
    ax_acc.legend()
    fig.savefig(image_file1)
    # model 2
    a,b,c= train(i,mode = "train",learning_rate = learning_rate, batch_size = batch_size, start_filters = start_filters, 
                 dropout_ratio = dropout_ratio, num_epochs = num_epochs, patience = patience)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))
    ax_loss.plot(range(0,num_epochs), a, label="Train loss")
    ax_loss.plot(range(0,num_epochs), b, label="Validation loss")
    ax_loss.legend()
    ax_acc.plot(range(0,num_epochs), c, label="Train accuracy")
    ax_acc.legend()
    fig.savefig(image_file2)

