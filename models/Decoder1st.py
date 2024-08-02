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
logging.basicConfig(filename = "Decoder_LSTM.log", filemode='w', level = logging.DEBUG)
random.seed('a')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42) 
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Decoder1st(nn.Module):
    def __init__(self, opt):
        super(Decoder1st, self).__init__()
        self.hidden_dim = opt.hidden_dim
        self.layer_dim = opt.layer
        self.r_shift = 288
        self.h_style_hr = 1
        self.cnn_dim = opt.cnn_dim
        
        self.lstm00 = nn.LSTM(1, self.hidden_dim, self.layer_dim, bidirectional=False, batch_first=True)
        self.lstm01 = nn.LSTM(1, self.hidden_dim, self.layer_dim, bidirectional=False, batch_first=True)
        self.lstm02 = nn.LSTM(1, self.hidden_dim, self.layer_dim, bidirectional=False, batch_first=True)
        self.lstm03 = nn.LSTM(1, self.hidden_dim, self.layer_dim, bidirectional=False, batch_first=True)

        self.L_out00 = nn.Linear(self.hidden_dim, 1)
        self.L_out03 = nn.Linear(self.hidden_dim, 1)

        self.cnn00 = nn.Conv1d(self.hidden_dim, self.cnn_dim, 7, stride=1, padding=3)
        self.cnn01 = nn.Conv1d(self.cnn_dim, 1, 3, stride=1, padding=1)

    def forward(self, x1, x2, x3, encoder_h, encoder_c):  # x1: time sin & cos;   x2: hinter vectors

        h0 = encoder_h[0]  # Far
        c0 = encoder_c[0]
        h1 = encoder_h[1]  # Ind
        c1 = encoder_c[1]
        h2 = encoder_h[2]  # Near
        c2 = encoder_c[2]
        sig = nn.Sigmoid()
        m = nn.Softmax(dim=1)
        x = x1     
        
        self.lstm00.flatten_parameters()
        self.lstm01.flatten_parameters()
        self.lstm02.flatten_parameters()
        self.lstm03.flatten_parameters()
        
        # Far Polar EDDU half-Output  
        o0, (hn, cn) = self.lstm00(torch.flip(x2, [1]), (h0,c0))
        out00 = self.L_out00(o0)
        out0 = torch.squeeze(out00, dim=2)            
        
        # Near Polar EDDU half-Output    
        o3, (hn, cn) = self.lstm03(torch.flip(x2, [1]), (h2,c2))
        out03 = self.L_out03(o3)
        out3 = torch.squeeze(out03, dim=2)        

        # Auxiliary EDDU half-Output
        o1, (hn, cn) = self.lstm01(torch.flip(x2, [1]), (h1,c1))           
        out = o1.permute(0, 2, 1)  
        cnn_out = self.cnn00(out)
        cnn_out = F.relu(self.cnn01(cnn_out))
        Ind = cnn_out.permute(0, 2, 1)
        Ind = torch.squeeze(Ind, dim=2)  
        
        return out0, Ind, out3

