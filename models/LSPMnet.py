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
from models.EncoderLSTM import *
from models.Decoder1st import *
from models.Decoder2nd import *
from utils.utils2 import *
from sklearn.metrics import mean_absolute_percentage_error
import logging
logging.basicConfig(filename = "LSPMnet.log", filemode='w', level = logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42) 
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LSPMnet(nn.Module):
    ENCODER_BLOCK = 'encoder'
    DECODER_1st = 'decoder1'
    DECODER_2nd = 'decoder2'

    def __init__(self,
                 opt,
                 device=torch.device('cuda'),
                 stack_types=(ENCODER_BLOCK, DECODER_1st, DECODER_2nd)
                 ):
        super(LSPMnet, self).__init__()
        self.opt = opt
        self.stack_types = stack_types
        self.stacks = []
        self.parameters = []
        self.device = device
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self.to(self.device)
        self._loss = None
        self._opt = None

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        blocks = []
        block_init = LSPMnet.set_block(stack_type)
        block = block_init(self.opt)
        self.parameters.extend(block.parameters())
        
        return block

    def save(self, filename: str):
        torch.save(self, filename)

    @staticmethod
    def set_block(block_type):
        if block_type == LSPMnet.ENCODER_BLOCK:
            return EncoderLSTM
        elif block_type == LSPMnet.DECODER_1st:
            return Decoder1st
        else:
            return Decoder2nd


    def forward(self, x1, x2, x3, h0, c0):

        # execute encoder
        h, c = self.stacks[0](x1, x2, x3, h0, c0)
        
        # execute decoder1st
        o0, Ind, o1 = self.stacks[1](x1, x2, x3, h, c)    

        # execute decoder2nd
        out0, out1, Ind, o4 = self.stacks[2](x1, o0, Ind, o1, h, c) 
        
        # ultimate output for prediction
        out3 = o4 + Ind  


        return out0, out1, Ind, out3

