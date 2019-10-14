import pandas as pd
import numpy as np
import csv
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os
import re

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_protein
from Bio import SeqIO

from transSeq import *
from utils import labels_dic, count_label

from sklearn.preprocessing import LabelBinarizer

np.set_printoptions(threshold=np.inf)


class prepareData():

    def __init__(self, type_name):
        self.type_name = type_name
        
        if self.type_name=='HA':
            self.label_1 = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9']
            self.label_2 = ['H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 'H17', 'H18']
            self.path = './data/HA-protein-IVD.fa'
        
        elif self.type_name=='NA':
            self.label_1 = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9']
            self.label_2 = ['N10', 'N11']
            self.path = './data/NA-protein-IVD.fa'
            
        self.max_len = 0
        self.num_acid = 0
        self.label_num_dic = []
        
    # read the initial sequences and select those after 2010 
    # the initial sequences form is: id; host; country; y/m/d; strain; segname; subtype
    def readFas(self, file_name):
        sequences = []
        label = []
        year = []
        for record in SeqIO.parse(file_name,"fasta"):
            desp = record.description.split('; ')
            # discard those incorect sequences
            if desp[-2]!=self.type_name:
                print(record.id, ' is excluded!')
                continue
                
            else:
                if self.type_name=='HA':
                    l_2 = desp[-1][:2]
                    l_3 = desp[-1][:3]
                elif self.type_name=='NA':
                    l_2 = desp[-1][-2:]
                    l_3 = desp[-1][-3:]
                # print('l_2 is', l_2)
                # print('l_3 is', l_3)
                y = desp[-3].split('/')
                if re.match(r'201\d', y[-1][:4]):
                    # append all sequences after 2010 (containing those number under 50) 
                    if l_3 in self.label_2:
                        year.append(y[-1][:4])
                        label.append(l_3)
                        sequences.append(record.seq)
                    elif l_2 in self.label_1:
                        year.append(y[-1][:4])
                        label.append(l_2)
                        sequences.append(record.seq)
                    else:
                        continue
                    
                else:
                    continue
        # calculate the number of each subtype (for all subtype)
        self.label_num_dic = count_label(label)
        return sequences, label, year
    
    # encode all seqs by one-hot
    def transfer(self, seqs_list):
        ts = TransSeq()
        final_seqs, max_len, num_acid = ts.trans(seqs_list)
        self.max_len = max_len
        self.num_acid = num_acid
        return final_seqs
    
    
    def select(self): 
        base_path = "./data_set/"
        sequences, labels, years = self.readFas(self.path)
        # print(len(sequences), ' sequences in total')
        # print(len(labels), ' labels in total')
        # print(len(years), 'years in total')
    
        label_dic = labels_dic(labels)
        # print('label_dic is', labels_dic)
        sequences_encoded = self.transfer(sequences)
        # print('The shape of seqs is ', sequences_encoded.shape)
        ds_path = base_path+str(self.type_name)+"/"+"data_label_"+str(self.type_name)+"_IVD.npz"
        np.savez(ds_path, sequences_encoded, labels, years, self.max_len, self.num_acid, self.label_num_dic)
        
        return ds_path
        
                        
                
                    
            
    
    
    
    
