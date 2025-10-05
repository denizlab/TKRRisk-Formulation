
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

from dataloader import image_loader,Radiographloader,MOSTloader

from collections import Counter, defaultdict

import argparse

import pickle

import pandas as pd
from helper import evaluate_all_base,evaluate_siam

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

def roc_auc_compute_fn(y_preds, y_targets):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")
    y_pred = y_preds.numpy()
    y_true = y_targets.numpy()

    #print(y_pred)
    #print(y_true)

    #print(y_pred.shape,y_true.sum())
    
    return roc_auc_score(y_true, y_pred)



class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="train a network for action recognition"
    )
    parser.add_argument("--config", type=str, help="path to a config file")
    parser.add_argument("--result_path", type=str, default=None,help="path to a model file")
    parser.add_argument("--dataset", type=str, help="MOST or OAI")
    parser.add_argument("--metric", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--cv", type=int)
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
    #result_path = args.result_path

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    print(device)
    seed = 1001
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

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

    cv = args.cv

    imb_mode = config.imb_mode

    if args.result_path is None:

        path = os.path.dirname(config_path)

        result_path = path + "/"+config.year+"/"+config.tl_model+"_model_"+"epochs_"+str(config.num_epoch)+"_lr_"+str(config.learning_rate)+"_ntimes_"+str(config.n_times)+"_layer_dim_"+str(config.layer_dim)+"_hidden_dim_"+str(config.hidden_dim)+"_imbtype_"+config.imb_mode+"_weight_decay_"+str(config.weight_decay)+"_setting_"+str(config.setting)+"_type_"+str(config.types)+"_gamma_"+str(config.gamma)+"_margin_"+str(config.margin)+"_pretrained_"+str(config.pretrained)+"/fold_"+str(fold)+"/CV_"+str(args.cv)+"/"
    else:
        result_path = args.result_path


    csv_path = "../../csv_files/Radiograph/"







    model= get_model(tl_model,mode,image_size,cropped_size,hidden_dim,layer_dim,dropout,output_dim,device,types=types).to(device)

    
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if args.dataset == "OAI":

        if args.mode != "test":
            test_df = pd.read_csv(csv_path+"Fold_"+str(fold)+"/CV_"+str(cv)+"_"+args.mode+".csv",index_col=0)

        else:
            test_df = pd.read_csv(csv_path+"Fold_"+str(fold)+"/Fold_"+str(fold)+"_"+args.mode+".csv",index_col=0)
        test_data = Radiographloader(test_df, image_size,cropped_size,last_index=2,mode='val',setting=setting,types=types,year = year)
    else:
        test_df = pd.read_csv("/gpfs/data/denizlab/Users/hrr288/data/reg_splits/Most_cohort.csv",index_col=0)
        test_data = MOSTloader(test_df, image_size,cropped_size,last_index=2,mode='val',setting=setting,types=types,year = year)


    
    test_loader = DataLoader(test_data, batch_size=8,shuffle=False)



    if types == "siamese":
        model.load_state_dict(torch.load(result_path+"best_"+args.metric+"_model.prm"))
        _,_,predictions_0_1,truths_0,predictions_1,truths = evaluate_siam(model,test_loader,device)
        pred_0 = torch.tensor(predictions_0_1).cpu().numpy()
        pred_1 = torch.tensor(predictions_1).cpu().numpy()
        labs  = torch.tensor(truths).cpu().numpy()
        test_df["truth"] = labs
        test_df["Pred_0"] = pred_0
        test_df["Pred_1"] = pred_1
        predictions_1.extend(predictions_0_1)
        truths.extend(truths_0)

        #all_auc = roc_auc_compute_fn(torch.sigmoid(torch.tensor(predictions_1)),torch.tensor(truths))

    else:
        model.load_state_dict(torch.load(result_path+"best_"+args.metric+"_model.prm"))

        _,predictions_0_1,truths_0,predictions_1,truths = evaluate_all_base(model,test_loader,device)
        pred_0 = torch.tensor(predictions_0_1).cpu().numpy()
        pred_1 = torch.tensor(predictions_1).cpu().numpy()
        labs  = torch.tensor(truths).cpu().numpy()
        test_df["truth"] = labs
        test_df["Pred_0"] = pred_0
        test_df["Pred_1"] = pred_1
        predictions_1.extend(predictions_0_1)
        truths.extend(truths_0)

        #all_auc = roc_auc_compute_fn(torch.sigmoid(torch.tensor(predictions_1)),torch.tensor(truths))
    #print(args.dataset," test overall AUC is ",all_auc)
    test_df.to_csv(result_path+"result_"+args.mode+"_"+args.dataset+"_"+args.metric+"_cv_"+str(args.cv)+"_df.csv")







   

if __name__ == "__main__":
    main()

