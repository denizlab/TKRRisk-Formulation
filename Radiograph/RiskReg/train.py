
from __future__ import print_function, division
import comet_ml
from comet_ml import Experiment
import os
import warnings
import numpy as np
import copy
import math
import scipy.ndimage as ndimage

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import XrayDataLoader
from torch.utils.data import DataLoader, Dataset

from dataloader import image_loader,Radiographloader

from collections import Counter, defaultdict

import argparse

import pickle

import pandas as pd
from helper import train,train_all_base,evaluate,train_siam,evaluate_siam

import random
import warnings
import re
from copy import deepcopy
import numpy as np
import pandas as pd
from model import get_model
import argparse
import torch.optim as optim

from config import get_config
import torch.nn.functional as F

from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout,BCEWithLogitsLoss,Dropout
from torch.optim import Adam, SGD
import warnings
warnings.filterwarnings('ignore')
from torch.autograd import Variable




def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="train a network for action recognition"
    )
    parser.add_argument("--config", type=str, help="path to a config file")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="a number used to initialize a pseudorandom number generator.",
    )

    return parser.parse_args()
def main() -> None:
    # argparser
    args = get_arguments()

    config_path = args.config

    config = get_config(config_path)
    path = os.path.dirname(config_path)

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    seed = 1001
    random.seed(seed)

    margin = config.margin
    cropped_size = config.cropped_size
    image_size = config.image_size
    gamma = config.gamma
    tl_model=config.tl_model
    output_dim = config.output_dim
    hidden_dim = config.hidden_dim
    batch_size = config.batch_size
    mode = config.mode
    layer_dim = config.layer_dim
    n_times = config.n_times
    learning_rate=config.learning_rate
    num_epoch = config.num_epoch
    dropout = config.dropout
    weight_decay = config.weight_decay
    setting = config.setting
    pretrained = config.pretrained
    types = config.types
    fold = config.fold
    year = config.year

    imb_mode = config.imb_mode


    csv_path = "../../csv_files/Radiograph/"





    
    for cv in range(1,7):
        result_path = path + "/"+config.year+"/"+config.tl_model+"_model_"+"epochs_"+str(config.num_epoch)+"_lr_"+str(config.learning_rate)+"_ntimes_"+str(config.n_times)+"_layer_dim_"+str(config.layer_dim)+"_hidden_dim_"+str(config.hidden_dim)+"_imbtype_"+config.imb_mode+"_weight_decay_"+str(config.weight_decay)+"_setting_"+str(config.setting)+"_type_"+str(config.types)+"_gamma_"+str(config.gamma)+"_margin_"+str(config.margin)+"_pretrained_"+str(config.pretrained)+"/fold_"+str(fold)+"/CV_"+str(cv)+"/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        train_df = pd.read_csv(csv_path+"Fold_"+str(fold)+"/CV_"+str(cv)+"_train.csv",index_col=0)
        val_df = pd.read_csv(csv_path+"Fold_"+str(fold)+"/CV_"+str(cv)+"_val.csv",index_col=0)

        
        model_name = config.year+config.tl_model+"_model_"+"epochs_"+str(config.num_epoch)+"_lr_"+str(config.learning_rate)+"_ntimes_"+str(config.n_times)+"_layer_dim_"+str(config.layer_dim)+"_hidden_dim_"+str(config.hidden_dim)+"_imbtype_"+config.imb_mode+"_mode_"+config.mode+"_weight_decay_"+str(config.weight_decay)+"_setting_"+str(config.setting)+"_type_"+str(config.types)+"_gamma_"+str(config.gamma)+"_margin_"+str(config.margin)+"_pretrained_"+str(config.pretrained)+"_fold_"+str(fold)+"_CV_"+str(cv)
        experiment = Experiment(api_key='Tft6eW3amg6SuIQfsV9RreOBa',project_name='ProgReg_nescv_expts')
        experiment.set_name(model_name)


        
        train_data = Radiographloader(train_df,image_size,cropped_size,last_index=2,mode='train',setting=setting,types=types,year = year)
        val_data = Radiographloader(val_df, image_size,cropped_size,last_index=2,mode='val',setting=setting,types=types,year = year)
        train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size,shuffle=False)


        model= get_model(tl_model,mode,image_size,cropped_size,hidden_dim,layer_dim,dropout,output_dim,device,types=types).to(device)

     
        




        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=int(len(train_loader)),epochs=num_epoch,final_div_factor = 1,anneal_strategy='linear')
        if types == "riskreg":
            loss_fn = nn.MarginRankingLoss(margin=margin).to(device)
            loss_fn_1 = BCEWithLogitsLoss().to(device)
            train_siam(train_loader, val_loader,config.year,val_df,model,optimizer,scheduler,loss_fn,loss_fn_1,experiment,gamma,result_path,dropout=dropout,num_epoch=num_epoch,device=device)
        else:
            loss_fn = BCEWithLogitsLoss(pos_weight=None).to(device)
            train_all_base(train_loader, val_loader,model,optimizer,scheduler,loss_fn,experiment,result_path,dropout=dropout,num_epoch=num_epoch,device=device)
        experiment.end()






   

if __name__ == "__main__":
    main()

