import numpy as np
from utils import *

from tkinter import _flatten
from data_processing import prepareData

np.set_printoptions(threshold=np.inf)

class data_set():
    def __init__(self, type_name):
        self.type_name = type_name
        if self.type_name=='HA':
            self.label_1 = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9']
            self.label_2 = ['H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 'H17', 'H18']
            
        elif self.type_name=='NA':
            self.label_1 = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9']
            self.label_2 = ['N10', 'N11']
        
        self.train_label_dic = {}
        self.max_len =0
        self.num_acid =0
        
        # self.path=ds_path

    def load_data(self):
        tr_label = []
        
        # load 
        data = np.load("./data_set/"+str(self.type_name)+"/"+"data_label_"+str(self.type_name)+"_IVD.npz")
        sequences = data["arr_0"]
        # print('length of sequences is', len(sequences))
        labels = data["arr_1"]
        self.all_label_dic = labels_dic(labels)
        # print('length of labels is', len(labels))
        year = data["arr_2"]
        max_len = data["arr_3"]
        print('max_len is', max_len)
        num_acid = data["arr_4"]
        print('num_acid is', num_acid)
        self.label_num_dic = data["arr_5"]
        
        
        # the seqs whose number is under 50 as new type
        for i in range(len(self.label_num_dic)):
            if self.label_num_dic[i]>=50:
                tr_label.append(get_keys(self.all_label_dic, i))
        self.train_label_dic = labels_dic(tr_label)
        print('label_dic above 50 is ', self.train_label_dic)
        
        return sequences, labels, year, max_len, num_acid
    
    # select seqs of each label
    def SeqofLable(self, seqs_list, labels_list, year_list=None):
        
        seqsoflabel = [[] for i in range(len(self.train_label_dic))]
        yearsoflabel = [[] for i in range(len(self.train_label_dic))]     
        # print('initial seqsoflabel is ', seqsoflabel)
        print('len of seqsoflabel is ', len(seqsoflabel))
        # add seq index to seqsoflabel, add year to yearsoflabel
        for i in range(len(seqs_list)):
            seqsoflabel[labels_list[i]].append(i)
            if year_list!=None:
                yearsoflabel[labels_list[i]].append(int(year_list[i]))
        print('seqsoflabel[0] shape is ', len(seqsoflabel[0]))
        print('seqsoflabel[1] shape is ', len(seqsoflabel[1]))
        print('yearsoflabel[0] shape is ', len(yearsoflabel[0]))
        print('yearsoflabel[1] shape is ', len(yearsoflabel[1]))
        
        return seqsoflabel, yearsoflabel
    
    def rank_as_y(self, seqs, labels, year):
        
        seqsoflabel, yearsoflabel = self.SeqofLable(seqs, labels, year)
        
        # for each subtype, rank seqs acoording year
        for i in range(len(yearsoflabel)):
            seqsoflabel[i] = np.array(seqsoflabel[i])
            index = np.argsort(yearsoflabel[i])
            # print('index length is', len(index))
            # print('index is', index)
            
            seqsoflabel[i] = seqsoflabel[i][index]
            
        return seqsoflabel
        
    # form train and test data set
    def split_dataset(self, seqs, label, year, rows, cols):
        
        
        seqs_train = []
        label_train = []
        year_train = []
        x_tr = []
        y_tr = []
        x_te = []
        y_te = []
        train_seqsoflabel = [[] for i in range(len(self.train_label_dic))]
        # num<50, then as new type in test set
        for i in range(len(seqs)):
            if self.label_num_dic[self.all_label_dic[label[i]]]<50:
                x_te.append(seqs[i])
                y_te.append(-1)
            else:
                seqs_train.append(seqs[i])
                label_train.append(self.train_label_dic[label[i]])
                year_train.append(year[i])  
        
        # rank the seqs except for whose num<50
        train_seqsdexoflabel = self.rank_as_y(seqs_train, label_train, year_train)
        
        # for each subtype, get the first 80% as train data and the rest as test data
        for i in range(len(train_seqsdexoflabel)):
            for j in range(len(train_seqsdexoflabel[i])):
                train_seqsoflabel[i].append(seqs_train[train_seqsdexoflabel[i][j]])
                if j<=int(0.8*len(train_seqsdexoflabel[i])):
                    x_tr.append(train_seqsoflabel[i][j])
                    y_tr.append(i)
                else:
                    x_te.append(train_seqsoflabel[i][j])
                    y_te.append(i)
              
        x_tr = np.array(x_tr)
        x_te = np.array(x_te)
        y_tr = np.array(y_tr)
        y_te = np.array(y_te)
        
        x_tr, y_tr = shuffle(x_tr, y_tr)
        x_te, y_te = shuffle(x_te, y_te)
        
        x_train = x_tr.reshape(x_tr.shape[0], rows*cols)
        x_test = x_te.reshape(x_te.shape[0], rows*cols)
    
    
        print(x_train.shape[0], ' train data in total')
        print(x_test.shape[0], ' test data in total')
        # print('a train sample is ', x_train[0])
        # print('a test sample is ', x_test[0])
        print('train label is ', y_tr[0])
        print('test label is', y_te[0])
        # print('tiny test label is', y_te_tiny[9])
        print('x_train shape is ', x_train.shape)
        print('x_test shape is ', x_test.shape)
        # print('x_tiny shape is', x_test_tiny.shape)
        
        return (x_train, y_tr), (x_test, y_te)

    
        
    def form_dataset(self):
        base_path = "./data_set/"+str(self.type_name)+"/"
        
        seqs, labels, years, self.max_len, self.num_acid = self.load_data()
        
        
        (x_train, y_train), (x_test, label_test) = self.split_dataset(seqs, labels, years, self.max_len, self.num_acid)
        # np.savez(base_path+"multi_train_8:2.npz", x_train, y_train, self.max_len, self.num_acid)
        # np.savez(base_path+"multi_test_8:2.npz", x_test, label_test, self.max_len, self.num_acid)
        
        
        seqsoflabel, _ = self.SeqofLable(x_train, y_train)
        
        # form negtive sample for each subtype
        for i in range(len(seqsoflabel)):
            negative = []
            positive = seqsoflabel[i]
            pos_num = len(seqsoflabel[i])
            for j in range(len(seqsoflabel)):
                if j==i:
                    continue
                else:
                    neg_rate = len(seqsoflabel[j])/(x_train.shape[0]-pos_num)
                    neg_num_j = int(neg_rate*pos_num)
                    
                    if neg_num_j>len(seqsoflabel[j]):
                        neg_slice = seqsoflabel[j]
                    else:
                        neg_slice = random.sample(seqsoflabel[j], neg_num_j)
                negative.append(neg_slice)
                negative = list(_flatten(negative))
            
            print('len of positive is', len(positive))
            print('len of negative is', len(negative))
            positive_seqs = x_train[positive]
            negative_seqs = x_train[negative]
            true_label = get_keys(self.train_label_dic ,i)
            print('true label is ', true_label)
            print('the shape of positive_seqs is ', positive_seqs.shape)
            print('the shape of negative_seqs is ', negative_seqs.shape)
        
        
            np.savez(base_path+"train_"+true_label+"_8:2.npz", positive_seqs, negative_seqs, self.max_len, self.num_acid, true_label)
         
        np.savez(base_path+"test_"+str(self.type_name)+"_8:2.npz", x_test, label_test, self.max_len, self.num_acid)
        
