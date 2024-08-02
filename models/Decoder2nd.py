#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import time,os,sys
import math

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
from utils.utils2 import *
from sklearn.metrics import mean_absolute_percentage_error
import logging
logging.basicConfig(filename = "Residue_LSTM.log", filemode='w', level = logging.DEBUG)
random.seed('a')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42) 
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Decoder2nd(nn.Module):
    def __init__(self, opt):
        super(Decoder2nd, self).__init__()
        self.hidden_dim = opt.hidden_dim
        self.layer_dim = opt.layer
        self.cnn_dim = opt.cnn_dim
        
        self.lstm00 = nn.LSTM(2, self.hidden_dim, self.layer_dim, bidirectional=False, batch_first=True) 
        self.lstm01 = nn.LSTM(2, self.hidden_dim, self.layer_dim, bidirectional=False, batch_first=True) 
        self.lstm02 = nn.LSTM(2, self.hidden_dim, self.layer_dim, bidirectional=False, batch_first=True) 

        self.L_out00 = nn.Linear(self.hidden_dim, 1)
        self.L_out01 = nn.Linear(self.hidden_dim, 1)
        self.L_out02 = nn.Linear(self.hidden_dim, 1)
        
        self.L_out04 = nn.Linear(2, self.cnn_dim)
        self.L_out05 = nn.Linear(self.cnn_dim, 1)       
        

    def forward(self, x1, out0, Ind, out3, encoder_h, encoder_c):  # x1: time sin & cos
        
        # Initialize hidden and cell state with zeros
        h0 = encoder_h[0]  # Far
        c0 = encoder_c[0]
        h1 = encoder_h[1]  # Ind
        c1 = encoder_c[1]
        h2 = encoder_h[2]  # Near
        c2 = encoder_c[2]
        sig = nn.Sigmoid()
        m = nn.Softmax(dim=1)  
        
        self.lstm00.flatten_parameters()
        self.lstm01.flatten_parameters()
        self.lstm02.flatten_parameters()
        
        # Polar EDDU Far-End Output 
        o0, (hn, cn) = self.lstm00(x1, (h0,c0))
        out00 = self.L_out00(o0)
        out00 = out0 + torch.squeeze(out00, dim=2)           
        
        # Polar EDDU Near-End Output        
        o3, (hn, cn) = self.lstm02(x1, (h2,c2))
        out03 = self.L_out02(o3)
        out03 = out3 + torch.squeeze(out03, dim=2)         

        # Auxiliary EDDU Final Output       
        o2, (hn, cn) = self.lstm01(x1, (h1,c1))
        out02 = self.L_out01(o2)
        Ind = Ind + torch.squeeze(out02, dim=2)        
        
        # Polar EDDUs Final output
        out00 = torch.unsqueeze(out00, dim=2)
        out03 = torch.unsqueeze(out03, dim=2)
        out02 = torch.cat([out00, out03], dim = 2)
        out4 = F.relu(self.L_out04(out02))
        out4 = self.L_out05(out4)  
        out4 = torch.squeeze(out4, dim=2)  
        
        return out0, out3, Ind, out4
