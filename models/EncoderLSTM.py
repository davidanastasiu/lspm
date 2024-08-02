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
logging.basicConfig(filename = "Encoder_LSTM.log", filemode='w', level = logging.DEBUG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42) 
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class EncoderLSTM(nn.Module):
    def __init__(self, opt):
        super(EncoderLSTM, self).__init__()        
        self.hidden_dim = opt.hidden_dim
        self.layer_dim = opt.layer
        self.cnn_dim = opt.cnn_dim
        
        """
        LSTM block. It does position embedding from 1-d input sequence to high dimensional hidden state.    
        """ 
        
        # far points
        self.lstm0 = nn.LSTM(1, self.hidden_dim, self.layer_dim, bidirectional=False, batch_first=True)
        # indicator
        self.lstm1 = nn.LSTM(1, self.hidden_dim, self.layer_dim, bidirectional=False, batch_first=True)
        # near points
        self.lstm2 = nn.LSTM(1, self.hidden_dim, self.layer_dim, bidirectional=False, batch_first=True)
        
    def forward(self, x1, x2, x3, h, c):

        self.lstm0.flatten_parameters()
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        
        out, (hn0, cn0) = self.lstm0(x3, (h,c)) # Far        
        out, (hn1, cn1) = self.lstm1(x3, (h,c)) # Ind        
        out, (hn2, cn2) = self.lstm2(x3, (h,c)) # Near
        
        hn = []
        cn = []
        hn.append(hn0) # Far
        hn.append(hn1) # Ind
        hn.append(hn2) # Near
        cn.append(cn0)
        cn.append(cn1)
        cn.append(cn2)


        
        return hn, cn

