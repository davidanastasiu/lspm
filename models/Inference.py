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
from models.LSPMnet import *
from models.EncoderLSTM import *
from models.Decoder1st import *
from models.Decoder2nd import *
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime, timedelta
import zipfile
import logging
logging.basicConfig(filename = "Inference.log", filemode='w', level = logging.DEBUG)
random.seed('a')
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42) 
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LSPM_I:

    def __init__(self, opt):

        self.logger = logging.getLogger()
        self.logger.info("I am logging...")
        self.opt = opt
        self.sensor_id = opt.stream_sensor       
        self.train_days = opt.input_len
        self.predict_days = opt.output_len  
        self.output_dim = opt.output_dim
        self.hidden_dim = opt.hidden_dim
        self.is_watersheds = 1
        self.is_prob_feature = 1
        self.TrainEnd = opt.model
        self.opt_hinter_dim = 1

        self.batchsize = opt.batchsize
        self.epochs = opt.epochs
        self.layer_dim = opt.layer

        self.net = LSPMnet(self.opt)
        self.optimizer = torch.optim.Adam(self.net.parameters(),self.opt.learning_rate)  

        self.criterion = nn.MSELoss(reduction='sum')

        self.expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.val_dir = os.path.join(self.opt.outf, self.opt.name, 'val')
        self.test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')        

    def std_denorm_dataset(self, predict_y0):
        
        pre_y = []
        a2 = log_std_denorm_dataset(self.mean, self.std, predict_y0, pre_y)

        return a2

    def inference_test(self, x_test, y_input1, y_input2):     
        
        y_predict = []
        d_out = torch.tensor([]).to(device)
        self.net.eval()

        with torch.no_grad():
            
            x_test = torch.from_numpy(np.array(x_test, np.float32)).to(device)
            y_input1 = torch.from_numpy(np.array(y_input1, np.float32)).to(device)
            y_input2 = torch.from_numpy(np.array(y_input2, np.float32)).to(device)

            h0 = torch.zeros(self.layer_dim, x_test.size(0),self.hidden_dim).to(device)
            c0 = torch.zeros(self.layer_dim, x_test.size(0),self.hidden_dim).to(device)
            _, _, _, out3 = self.net(y_input1, y_input2, x_test, h0, c0)
            y_predict = [out3[0][i].item() for i in range(len(out3[0]))]
            y_predict = np.array(y_predict).reshape(1,-1) 
            
        return y_predict


    def model_load(self,zipf):       
        
        with zipfile.ZipFile(zipf, "r") as file:
            file.extract("Norm.txt")
        norm = np.loadtxt('Norm.txt',dtype=float,delimiter=None)
        os.remove('Norm.txt')
        print("norm is: ", norm)
        self.mean = norm[0]
        self.std = norm[1]
        self.R_mean = norm[2]
        self.R_std = norm[3]
         
        with zipfile.ZipFile(zipf, "r") as archive:
            with archive.open("LSPM.pt","r") as pt_file:
                self.net.load_state_dict(torch.load(pt_file), strict=False)                           

    def get_data(self, test_point):
        
        print("test_point is: ", test_point)
        # data prepare
        trainX = pd.read_csv('./data_provider/datasets/'+ self.opt.stream_sensor+'.csv', sep='\t')
        trainX.columns = ["id", "datetime", "value"] 
        trainX.sort_values('datetime', inplace=True),
        R_X = pd.read_csv('./data_provider/datasets/'+ self.opt.rain_sensor+'.csv', sep='\t')
        R_X.columns = ["id", "datetime", "value"] 
        R_X.sort_values('datetime', inplace=True)
        
        # read stream data        
        point = trainX[trainX["datetime"]==test_point].index.values[0]
        stream_data = trainX[point-15*24*4:point]["value"].values.tolist()
        gt = trainX[point:point+3*24*4]["value"].values.tolist()
        NN = np.isnan(stream_data).any() 
        if NN:
            print("There is None value in the stream input sequence.")  
        NN = np.isnan(gt).any() 
        if NN:
            print("There is None value in the ground truth sequence.")  
        
        # read rain data
        R_X = pd.read_csv('./data_provider/datasets/'+self.opt.rain_sensor+'.csv', sep='\t')
        R_X.columns = ["id", "datetime", "value"] 
        point = R_X[R_X["datetime"]==test_point].index.values[0]
        rain_data = R_X[point-288:point]["value"].values.tolist()
        NN = np.isnan(rain_data).any() 
        if NN:
            print("There is None value in the rain input sequence.")      
       
        return stream_data, rain_data, gt
    
    def test_single(self, test_point):
        
        stream_data, indicator_data, gt = self.get_data(test_point)  
        pre = self.predict(test_point, stream_data, indicator_data)
        
        return pre, gt
    
    def predict(self, test_point, stream_data, rain_data=None):
        
        time_str = test_point
        self.net.eval()
        test_predict = np.zeros(self.predict_days*self.output_dim)
                                
        test_month = []
        test_day = []
        test_hour = []
        new_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        for i in range(self.predict_days):
            new_time_temp = new_time + timedelta(minutes=15)
            new_time = new_time.strftime("%Y-%m-%d %H:%M:%S")
            
            month = int(new_time[5:7])
            day = int(new_time[8:10])
            hour = int(new_time[11:13])
   
            test_month.append(month)
            test_day.append(day)
            test_hour.append(hour)  
            
            new_time = new_time_temp
            
        y2 = cos_date(test_month, test_day, test_hour) 
        y2 = [[ff] for ff in y2]
 
        y3 = sin_date(test_month, test_day, test_hour) 
        y3 = [[ff] for ff in y3]        
        
        y_input1 = np.array([np.concatenate((y2,y3),1)])
       
        x_test = np.array(log_std_normalization_1(stream_data, self.mean, self.std), np.float32).reshape(self.train_days,-1)

        if rain_data is None:
            raise ValueError("Rain data is required.")
        y4 = np.array(log_std_normalization_1(rain_data, self.R_mean, self.R_std)).reshape(self.predict_days, -1)

        x_test = [x_test]
        y_input2 = [y4]
        y_predict = self.inference_test(x_test, y_input1, y_input2)
        y_predict = np.array(y_predict.tolist())[0]
        y_predict = [y_predict[i].item() for i in range(len(y_predict))]
        test_predict = np.array(self.std_denorm_dataset(y_predict))
        diff_predict = []
        test_predict = (test_predict + abs(test_predict))/2 
        
        return test_predict
    

