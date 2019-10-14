import keras
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils, plot_model, to_categorical


import numpy as np
import tensorflow as tf
from keras import backend as K
import sys

import time
import os
import pandas

from utils import *
from Resnet import Resnet


# seed = 7
# np.random.seed(seed)

np.set_printoptions(threshold=np.inf)
os.environ['KERAS_BACKEND'] = 'tensorflow'

   
class train():
    def __init__(self, args):
        self.type_name = args.type_name
        self.args = args
        self.min_loss = 100
        # self.multi = args.multi
         
    def run(self):
        sys.stdout = Logger("train_record.txt")
        start = time.clock()
        
        
        if self.type_name=='HA':
            self.l_name1 = ['H1', 'H3', 'H4', 'H5', 'H6', 'H7', 'H9', 'H10', 'H11', 'H2', 'H13', 'H16']
            self.l_name2 = ['H2', 'H13', 'H16']
            # self.num_classes = 12
        elif self.type_name=='NA':
            self.l_name1 = ['N1', 'N2', 'N3', 'N6', 'N7', 'N8', 'N9', 'N5']
            self.l_name2 = ['N5']
            # self.num_classes = 8
    
        # if self.multi==0:
        for l in self.l_name1:
            sequences, label, max_len, num_acid = load_dataset(l, self.type_name)
            # split train and valid data
            x_train, x_val, y_train, y_val = train_test_split(sequences, label, test_size=0.2)
            
            x_val = x_val.reshape(x_val.shape[0], max_len, num_acid, 1)
            x_train = x_train.reshape(x_train.shape[0], max_len, num_acid, 1)
            print('x_val shape is', x_val.shape)
            print('x_train shape is', x_train.shape)
    
        
            # make a history object
            history = LossHistory()
            resnet = Resnet(l, history, self.args)
            model = resnet.set_model(max_len, num_acid)
            
            if l in self.l_name2:
                model.load_weights('./result/'+self.type_name+'/weight/model_weights_'+self.best_l+'_8:2.h5')
            resnet.run(model, x_train, x_val, y_train, y_val)
            if l in self.l_name1:
                if self.min_loss>history.val_loss['epoch'][-1]:
                    self.min_loss = history.val_loss['epoch'][-1]
                    self.best_l = l
        print ('the training is ok!')
        end = time.clock()
        print('Time cost is:', end-start)
            
        
        history.acc_plot('epoch', l, self.type_name)
        history.loss_plot('epoch', l, self.type_name)
        '''        
        else:
                sequences, label, max_len, num_acid = load_multidata(self.type_name)
                # split train and valid data
                x_train, x_val, y_train, y_val = train_test_split(sequences, label, test_size=0.2)
            
                x_val = x_val.reshape(x_val.shape[0], max_len, num_acid, 1)
                x_train = x_train.reshape(x_train.shape[0], max_len, num_acid, 1)
                print('x_val shape is', x_val.shape)
                print('x_train shape is', x_train.shape)
                
                y_train = to_categorical(y_train, num_classes=self.num_classes)
                y_val = to_categorical(y_val, num_classes=self.num_classes)
    
                # make a history object
                history = LossHistory()
                resnet = Resnet('multi', history, self.args)
                model = resnet.set_multimodel(max_len, num_acid)
                
                resnet.run(model, x_train, x_val, y_train, y_val)
                print ('the training is ok!')
                end = time.clock()
                print('Time cost is:', end-start)
                
                history.acc_plot('epoch', 'multi', self.type_name)
                history.loss_plot('epoch', 'multi', self.type_name)
        '''
    

   