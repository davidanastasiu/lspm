import time,os,sys
import math

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils.utils2 import *
from models.LSPMnet import *
from models.EncoderLSTM import *
from models.Decoder1st import *
from models.Decoder2nd import *
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime, timedelta
import zipfile
import logging
from utils.metric import *
logging.basicConfig(filename = "LSPM.log", filemode='w', level = logging.DEBUG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42) 
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LSPM:

    def __init__(self, opt, dataset):

        self.logger = logging.getLogger()
        self.logger.info("I am logging...")
        self.dataset = dataset
        self.opt = opt
        self.sensor_id = opt.stream_sensor
        self.dataloader = dataset.get_train_data_loader()
        if self.opt.mode == 'train':
            self.val_data = np.array(dataset.get_val_points()).squeeze(1)
        self.trainX = dataset.get_trainX()
        self.data = dataset.get_data()
        self.sensor_data_norm = dataset.get_sensor_data_norm()
        self.sensor_data_norm_1 = dataset.get_sensor_data_norm1()
        self.R_norm_data = dataset.get_R_sensor_data_norm()
        self.mean = dataset.get_mean()
        self.std = dataset.get_std()
        self.month = dataset.get_month()
        self.day = dataset.get_day()
        self.hour = dataset.get_hour()
        
        self.train_days = opt.input_len
        self.predict_days = opt.output_len  
        self.output_dim = opt.output_dim
        self.hidden_dim = opt.hidden_dim
        self.is_watersheds = 1
        self.is_prob_feature = 1
        self.TrainEnd = opt.model
        self.opt_hinter_dim = 1
        self.os = 1
        self.r_shift = 288

        self.batchsize = opt.batchsize
        self.epochs = opt.epochs
        self.layer_dim = opt.layer

        self.net = LSPMnet(self.opt)
        self.optimizer = torch.optim.Adam(self.net.parameters(),self.opt.learning_rate)  

        self.criterion = nn.MSELoss(reduction='sum')

        self.expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.val_dir = os.path.join(self.opt.outf, self.opt.name, 'val')
        self.test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        
        self.train_loss_list = []
        self.val_loss_list = []
        
    def get_train_loss_list(self):
        
        return self.train_loss_list
    
    def get_val_loss_list(self):
        
        return self.val_loss_list


    def std_denorm_dataset(self, predict_y0, pre_y):
    
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


    def test_single(self, test_point):

        self.net.eval()

        test_predict = np.zeros(self.predict_days*self.output_dim)
                                
        #foot label of test_data
        point = self.trainX[self.trainX["datetime"]==test_point].index.values[0]
        start_num = self.trainX[self.trainX["datetime"]==self.opt.start_point].index.values[0]
        test_point = point - start_num
        pre_gt = self.trainX[point-1:point+self.opt.output_len-1]["value"].values.tolist()
        y = self.trainX[point:point+self.predict_days]["value"]

        b = test_point
        e = test_point+self.predict_days
     
        y2 = cos_date(self.month[b:e], self.day[b:e], self.hour[b:e]) # represent cos(int(data)) here
        y2 = [[ff] for ff in y2]
 
        y3 = sin_date(self.month[b:e], self.day[b:e], self.hour[b:e]) # represent sin(int(data)) here
        y3 = [[ff] for ff in y3]
        
        y_input1 = np.array([np.concatenate((y2,y3),1)])

        #inference
        norm_data = self.sensor_data_norm
        if(self.is_watersheds==1 or self.is_prob_feature==1):
            x_test = np.array(self.sensor_data_norm_1[test_point-self.train_days*1:test_point], np.float32).reshape(self.train_days,-1)
        else:
            x_test = np.array(norm_data[test_point-self.train_days*1:test_point], np.float32).reshape(self.train_days,-1)
        y_test = np.array(norm_data[test_point:test_point+self.predict_days], np.float32).reshape(self.predict_days,-1)
        y4 = np.array(self.R_norm_data[test_point-self.r_shift:test_point+self.predict_days-self.r_shift], np.float32).reshape(self.predict_days,-1)
        x_test = [x_test]
        y_input2 = [y4]
                
        y_predict = self.inference_test(x_test, y_input1, y_input2)
        y_predict = np.array(y_predict.tolist())[0]
        y_predict = [y_predict[i].item() for i in range(len(y_predict))]
    
        pre_gt = self.data[test_point-1:test_point]
        pre_gt = pre_gt[0]
        if pre_gt is None:
            print(pre_gt)

        test_predict = np.array(self.std_denorm_dataset(y_predict, pre_gt))
        test_predict = (test_predict + abs(test_predict))/2
        return test_predict, y

    def test_single_new(self, test_point, rain_data):
        
        time_str = test_point
        self.net.eval()
        test_predict = np.zeros(self.predict_days*self.output_dim)
                                
        #foot label of test_data
        point = self.trainX[self.trainX["datetime"]==test_point].index.values[0]
        start_num = self.trainX[self.trainX["datetime"]==self.opt.start_point].index.values[0]
        test_point = point - start_num


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

        #inference
        self.sensor_data_norm_1 = self.dataset.get_sensor_data_norm1()
        norm_data = self.dataset.get_sensor_data_norm()
        
        if(self.is_prob_feature==1):
            x_test = np.array(self.sensor_data_norm_1[test_point-self.train_days*1:test_point], np.float32).reshape(self.train_days,-1)
        else:
            x_test = np.array(norm_data[test_point-self.train_days*1:test_point], np.float32).reshape(self.train_days,-1)
        y4 = np.array((rain_data - self.dataset.get_R_mean())/self.dataset.get_R_std()).reshape(self.predict_days, -1)

        x_test = [x_test]
        y_input2 = [y4]
        y_predict = self.inference_test(x_test, y_input1, y_input2)
        y_predict = np.array(y_predict.tolist())[0]
        y_predict = [y_predict[i].item() for i in range(len(y_predict))]
        pre_gt = []
        test_predict = np.array(self.std_denorm_dataset(y_predict, pre_gt))
        diff_predict = []
        test_predict = (test_predict + abs(test_predict))/2
        return test_predict
    

    def generate_single_val_rmse(self, min_RMSE=500):
        
        total = 0 
        val_rmse_list = []
        val_pred_list = []
        val_pred_lists_print = []
        gt_mape_list = []
        val_mape_list = []
        val_points = self.val_data
        test_predict = np.zeros(self.predict_days * self.output_dim)

        non_flag = 0                        
        for i in range(len(val_points)):
            
            val_pred_list_print = []
            val_point = val_points[i]
            test_predict, ground_truth = self.test_single(val_point)  
            rec_predict = test_predict
            
            for j in range(len(rec_predict)):
                temp = [val_point, j, rec_predict[j]]
                val_pred_list.append(temp)
                val_pred_list_print.append(rec_predict[j])
            
            val_pred_lists_print.append(val_pred_list_print)
            val_MSE = np.square(np.subtract(ground_truth, test_predict)).mean() 
            val_RMSE = math.sqrt(val_MSE)
            val_rmse_list.append(val_RMSE)
            total += val_RMSE
            
            if (np.isnan(ground_truth).any()):
                print("val_point is: ", val_point)
                print("groud_truth:", ground_truth)
                non_flag = 1
            if (np.isnan(test_predict).any()):
                print("val_point is: ", val_point)
                print("there is non in test_predict:", test_predict)
                non_flag = 1
            gt_mape_list.extend(ground_truth)
            val_mape_list.extend(test_predict)
            
        name = "%s" % (self.opt.model)
        file_name = os.path.join(self.val_dir, 'validation_timestamps_24avg.tsv')        

        new_min_RMSE = min_RMSE

        if(self.is_watersheds == 1) :
            if(self.is_prob_feature == 0) :
                watersheds = "shed"
            else:
                watersheds = "Shed-ProbFeature"
        elif(self.is_prob_feature == 0) :
            watersheds = "solo"
        else:
            watersheds = "ProbFeature"
                
        basic_path = self.val_dir + '/' + str(self.sensor_id)
 
        
        if total < min_RMSE:

            aa = pd.DataFrame(data = val_pred_lists_print)
            aa.to_csv(basic_path + '_' + watersheds + str(self.TrainEnd) + '_pred_lists_print.tsv', sep = '\t')

            #save_model
            net_name = self.expr_dir + '/' + 'LSPM.pt'    
            new_min_RMSE = total        
            expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
            c_dir = os.getcwd()
            os.chdir(expr_dir)          
            with zipfile.ZipFile(self.opt.name+".zip", "w") as my_zip:
                with my_zip.open("LSPM.pt", "w") as data_file:
                    torch.save(self.net.state_dict(), data_file)
            os.chdir(c_dir)                
        print("val total RMSE: ", total)
        print("val min RMSE: ", new_min_RMSE)

        if non_flag == 0:
            mape = mean_absolute_percentage_error(np.array(gt_mape_list)+1, np.array(val_mape_list)+1)
        else:
            mape = 100
            
        return total, new_min_RMSE, mape
    
    def model_load(self):

        c_dir = os.getcwd()
        os.chdir(self.expr_dir) 
        model = LSPMnet(self.opt)
        with zipfile.ZipFile(self.opt.name +'.zip', "r") as archive:
            with archive.open("LSPM.pt","r") as pt_file:
                model.load_state_dict(torch.load(pt_file), strict=False)
                print("Importing the best pt file:", pt_file)
        os.chdir(c_dir)
        self.net = model
    
    def generate_test_rmse_mape(self):
        
        total = 0 
        val_rmse_list = []
        val_pred_list = []
        val_pred_lists_print = []
        gt_mape_list = []
        val_mape_list = []
        val_points = self.test_data
        test_predict = np.zeros(self.predict_days * self.output_dim)

        non_flag = 0     
        start = time.time()
        for i in range(len(val_points)):     
            start = time.time()
            val_pred_list_print = []
            val_point = val_points[i]
            test_predict, ground_truth = self.test_single(val_point)  
            rec_predict = test_predict
            val_MSE = np.square(np.subtract(ground_truth, test_predict)).mean() 
            val_RMSE = math.sqrt(val_MSE)
            val_rmse_list.append(val_RMSE)
            total += val_RMSE         
            
            for j in range(len(rec_predict)):
                temp = [val_point, j, rec_predict[j]]
                val_pred_list.append(temp)
                val_pred_list_print.append(rec_predict[j])
            
            val_pred_lists_print.append(val_pred_list_print)
            gt_mape_list.extend(ground_truth)
            val_mape_list.extend(test_predict)
        end = time.time()
        print("Inferencing test points ", len(val_points), " use: ", end-start)
        

        pd_temp = pd.DataFrame(val_pred_list, columns=("start", "No.", "prediction"))


        if(self.is_watersheds == 1) :
            if(self.is_prob_feature == 0) :
                watersheds = "shed"
            else:
                watersheds = "Shed-ProbFeature"
        elif(self.is_prob_feature == 0) :
            watersheds = "solo"
        else:
            watersheds = "ProbFeature"
                
        basic_path = self.test_dir + '/' + str(self.sensor_id)
        basic_model_path = self.expr_dir + '/' + str(self.sensor_id)   
        
        if self.opt.save == 1:
            aa = pd.DataFrame(data = val_pred_lists_print)
            i_dir = basic_path + '_' + watersheds + str(self.TrainEnd) + '_pred_lists_print.tsv'
            aa.to_csv(i_dir, sep = '\t')
            print("Inferencing result is saved in: ", i_dir)

        if non_flag == 0:
            mape = mean_absolute_percentage_error(np.array(gt_mape_list)+1, np.array(val_mape_list)+1)
        else:
            mape = 100
            
        return total,  mape,  val_pred_lists_print

    
    def inference(self): 
        start = time.time()
        self.dataset.gen_test_data() # generate test points file and test_data
        end = time.time()
        print("generate test points file and test_data: ", end-start)
        # refresh the related values
        self.test_data = np.array(self.dataset.get_test_points()).squeeze(1) # read the test set
        self.data = self.dataset.get_data()
        self.sensor_data_norm = self.dataset.get_sensor_data_norm()
        self.sensor_data_norm_1 = self.dataset.get_sensor_data_norm1()
        self.R_norm_data = self.dataset.get_R_sensor_data_norm()
        self.mean = self.dataset.get_mean()
        self.std = self.dataset.get_std()
        self.month = self.dataset.get_month()
        self.day = self.dataset.get_day()
        self.hour = self.dataset.get_hour()   
        rmse, mape, aa = self.generate_test_rmse_mape() # inference on test set
        return aa

    def compute_metrics(self, aa):
        val_set=pd.read_csv('./data_provider/datasets/test_timestamps_24avg.tsv',sep='\t')
        val_points=val_set["Hold Out Start"]
        trainX = pd.read_csv('./data_provider/datasets/'+self.opt.stream_sensor+'.csv', sep='\t')
        trainX.columns = ["id", "datetime", "value"] 
        count = 0
        for test_point in val_points:
            point = trainX[trainX["datetime"]==test_point].index.values[0]
            NN = np.isnan(trainX[point-15*24*4:point+3*24*4]["value"]).any() 
            if not NN:
                count += 1
        vals4 = aa
        # compute metrics
        all_GT=[]
        all_LSPM=[]
        loop = 0
        ind = 0
        while loop<len(val_points):
            ii=val_points[loop]
            val_point=val_points[ind]
            point=trainX[trainX["datetime"]==ii].index.values[0]
            x=trainX[point-15*24*4:point+3*24*4]["value"].values.tolist()    
            if (np.isnan(np.array(x)).any()):
                loop = loop + 1 # id for time list 
                continue
            loop = loop + 1
            if ind >= count - count%100:
                break   
            ind += 1  
            temp_vals4=list(vals4[ind-1])
            all_GT.extend(x[15*24*4:])
            all_LSPM.extend(temp_vals4) 
        metrics = metric_rolling(np.array(all_LSPM), np.array(all_GT))
        return metrics


    def train(self):

        num_epochs = self.epochs
        early_stop = 0
        old_val_loss = 1000
        min_RMSE = 500000
        sig = nn.Sigmoid()
        m = nn.Softmax(dim=1)
        
        for epoch in range(num_epochs):
            print_loss_total = 0  # Reset every epoch
            self.net.train()
            start = time.time()
            
            for i, batch in enumerate(self.dataloader):
                x_train = [TrainData for TrainData, _ in batch]
                y_train = [TrainLabel for _, TrainLabel in batch]
                y_train = np.array(y_train).squeeze()
                    
                y_train1 = []
                for ii in range(len(y_train)):
                    y_train1.append([yy for yy in [xx[:][1:3] for xx in y_train[ii]]]) 

                y_train0 = []
                for ii in range(len(y_train)):
                    y_train0.append([yy for yy in [xx[:][0] for xx in y_train[ii]]])    
                                        
                y_train2 = []
                for ii in range(len(y_train)):
                    y_train2.append([yy for yy in [xx[:][3:4] for xx in y_train[ii]]])
                                    
                x_train = torch.from_numpy(np.array(x_train, np.float32)).to(device)   # stream 15 days input
                y_train = torch.from_numpy(np.array(y_train0, np.float32)).to(device)  # stream gt
                decoder_input1 = torch.from_numpy(np.array(y_train1,np.float32)).to(device) # time feature
                decoder_input2 = torch.from_numpy(np.array(y_train2,np.float32)).to(device) # rain 3 days input


                seg_label_g = y_train          
                   
                self.optimizer.zero_grad()
                    
                loss = 0        
                
                h0 = torch.zeros(self.layer_dim, x_train.size(0), self.hidden_dim).to(device)
                c0 = torch.zeros(self.layer_dim, x_train.size(0), self.hidden_dim).to(device) 
                
                # Forward pass
                out0, out1, Ind, out3 = self.net(decoder_input1, decoder_input2, x_train, h0, c0) 

                #point weight
                weights0 = torch.tanh(y_train)**2    # tanh far
                weights1 = (1 - torch.abs(torch.tanh(y_train))) **2   # near

                # far representation loss
                loss1 = self.criterion(torch.mul(out0,weights0), torch.mul(y_train,weights0)) 
                
                # near representation loss
                loss3 = self.criterion(torch.mul(out1,weights1), torch.mul(y_train,weights1)) 
                
                # short-term specific loss
                loss2 = self.criterion(Ind[:,:6], y_train[:,:6])
    
                # pred_y
                loss5 = self.criterion(out3, y_train)
                
                # decay factor
                l_w = max(-1 * np.exp(epoch/45) + 2, 0.2)
                
                loss = l_w * (loss1 + loss3+ loss2) + loss5 

                loss.backward()
                self.optimizer.step()
                print_loss_total += loss.item()                    
        
        
            self.net.eval()
        
            val_loss, min_RMSE, mape = self.generate_single_val_rmse(min_RMSE)
            self.train_loss_list.append(print_loss_total)
            self.val_loss_list.append(val_loss)
            end = time.time()
            print('-----------Epoch: {}. train_Loss>: {:.6f}. --------------------'.format(epoch, print_loss_total)) 
            print('-----------Epoch: {}. val_Loss_rmse>: {:.6f}. --------------------'.format(epoch, val_loss)) 
            print('-----------Epoch: {}. val_Loss_mape>: {:.6f}. --------------------'.format(epoch, mape))
            print('-----------Epoch: {}. running time>: {:.6f}. --------------------'.format(epoch, end-start))
            self.logger.info('-----------Epoch: {}. train_Loss>: {:.6f}. --------------------'.format(epoch, print_loss_total))
            self.logger.info('-----------Epoch: {}. val_Loss_rmse>: {:.6f}. --------------------'.format(epoch, val_loss))
            self.logger.info('-----------Epoch: {}. val_Loss_mape>: {:.6f}. --------------------'.format(epoch, mape))
            self.logger.info(time.time())
            #early stop
            if(val_loss > min_RMSE):
                early_stop += 1
            else:
                early_stop = 0
            if(early_stop > 5 ):
                break
            old_val_loss = val_loss  
            
            adjust_learning_rate(self.optimizer, epoch + 1, self.opt)
                                

    

