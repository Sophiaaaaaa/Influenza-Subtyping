import random
import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.utils import to_categorical
import keras
import matplotlib.pyplot as plt

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_protein
from Bio import SeqIO

import re
import os
import sys


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))


    def on_epoch_end(self, epoch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
       
        
    def loss_plot(self, loss_type, l_name, type_name):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        # plt.plot(iters, self.accuracy[loss_type],'r',label = 'train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            # plt.plot(iters, self.val_acc[loss_type], 'b', label = 'val_acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val_loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.savefig('./result/'+type_name+'/loss/'+'loss_'+l_name+'.png')
        # plt.show()

    def acc_plot(self, loss_type, l_name, type_name):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        # plt.plot(iters, self.losses[loss_type], 'g', label = 'train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val_acc')
            # val_loss
            # plt.plot(iters, self.val_loss[loss_type], 'k', label = 'val_loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc')
        plt.legend(loc="upper right")
        plt.savefig('./result/'+type_name+'/acc/'+'acc_'+l_name+'.png')
        # plt.show()

    

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

  
def get_keys(dic, value):
    for k,v in dic.items():
        if v == value:
            return k
    
def shuffle(x, y):
    # print('the shape of x is', x.shape[0])
    index = [i for i in range(x.shape[0])]  
    random.shuffle(index)
    x = x[index]
    y = y[index]
    # print('the index is', index)
    # print('the shape of x after is', x.shape)
    return x, y
'''
def load_multidata(type_name, isTrain=True):
    if isTrain==True:
        data_name = "/multi_train_8:2.npz"
    else:
        data_name = "/multi_test_8:2.npz"
        
    data_path = "./data_set/"+str(type_name)+data_name
    data = np.load(data_path)
    sequences = data["arr_0"]
    label = data["arr_1"]
    print(sequences.shape[0], 'seqs for multi_class')
    print(label.shape[0], 'labels for multi_class')
    max_len = data["arr_2"]
    num_acid = data["arr_3"]        
    
    return sequences, label, max_len, num_acid
'''
def load_dataset(lname, type_name='HA', isTrain=True):
    label = []
    if isTrain==True:
        base_path = "./data_set/"+str(type_name)+"/train_"
        data = np.load(base_path+str(lname)+"_8:2.npz")
        positive_seqs = data["arr_0"]
        negative_seqs = data["arr_1"]
        print(positive_seqs.shape[0], 'positive seqs of ', lname)
        print(negative_seqs.shape[0], 'negative seqs of ', lname)
        max_len = data["arr_2"]
        num_acid = data["arr_3"]
        for i in range(positive_seqs.shape[0]):
            label.append(1)
        for i in range(negative_seqs.shape[0]):
            label.append(0)
        label = np.array(label)
        print('len of label is', label.shape)
        sequences = np.concatenate((positive_seqs,negative_seqs),axis=0)
        print('shape of sequences_concate is ', sequences.shape)
        sequences, label = shuffle(sequences, label)
    
    else:
        base_path="./data_set/"+str(type_name)+"/test_"
        data = np.load(base_path+str(type_name)+"_8:2.npz")
        sequences = data["arr_0"]
        label = data["arr_1"]
        max_len = data["arr_2"]
        num_acid = data["arr_3"]
        
    return sequences, label, max_len, num_acid


def labels_dic(labels_list):
        
    label_set = list(set(labels_list))
    label_set.sort()
    index = [i for i in range(len(label_set))]
    label_dic = dict(zip(label_set, index))
    return label_dic

# calculate the number of each lable
def count_label(labels_list):
    # {H1:0, H10:1,...}
    label_dic = labels_dic(labels_list)
    # print('label_dic is', label_dic)
    '''
    class_num = len(set(labels_list))
    print('class_num is', class_num)
    print('class contains', set(labels_list))
    '''
    label_num = [0 for i in range(len(label_dic))]
    for k,v in label_dic.items():
        label_num[v] = labels_list.count(k)
    print('label_num is', label_num)
    return label_num

    
def batch_size(l_name):
    if l_name=='H16' or l_name=='N5':
        batch_size = 4
    elif l_name=='H2' or l_name=='H11' or l_name=='H13':
        batch_size = 8
    elif l_name=='H4' or l_name=='H10' or l_name=='N3' or l_name=='N7':
        batch_size = 16
    elif l_name=='H6' or l_name=='H7' or l_name=='N6' or l_name=='N8' or l_name=='N9':
        batch_size = 32
    else:
        batch_size = 128
        
    return batch_size

def normalize(x_train, x_test):
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    
    x_train = (x_train - mean)/std
    x_test = (x_test - mean)/std
    
    return x_train, x_test
'''
def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    x_train, x_test = normalize(x_train, x_test)
    
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)
    
    return x_train, x_test, y_train, y_test
'''
   
def check_folder(path):
    path_new = path
    if not os.path.exists(path):
        os.mkdir(path)
    return path_new
    
