from Bio import SeqIO
import numpy as np
from sklearn.preprocessing import LabelBinarizer
# from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import sys
import re

np.set_printoptions(threshold=np.inf)

class TransSeq():
    def __init__(self):
        # amino acid of influenza A seqs
        self.coding = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X', 'B', 'J', 'Z']
        self.char_to_int = dict((c, i) for i, c in enumerate(self.coding))
        self.canda_vec = [0 for i in range(len(self.coding))]
     
    def convert_to_onehot(self):
        encoder = LabelBinarizer()
        onehot_encoded = encoder.fit_transform(self.coding)
        return onehot_encoded
    
    def max_len(self, seqs_list):
        max_len = 0
        for seq in seqs_list:
            if max_len<len(seq):
                max_len = len(seq)
        return max_len  
    
    def trans(self, seqs_list):
        max_len = self.max_len(seqs_list)
        print('max len is', max_len)
        # for each seq
        final_tmp_seqs = []
        for i in range(len(seqs_list)):
            sequence_code = []
            for char in seqs_list[i]:
                # get the one-hot coding of the aeq
                char_code = self.convert_to_onehot()[self.char_to_int[char]]
                sequence_code.append(char_code)
            
            if len(sequence_code)<max_len:
                for i in range(len(sequence_code), max_len):
                    sequence_code.append(self.canda_vec)
            
            sequence_coded = np.array(sequence_code)
            
            sequence_coded = sequence_coded.flatten()
            # print('the shape of sequence_coded is', sequence_coded.shape)
            
            # return one-hot coded seqs_list
            final_tmp_seqs.append(sequence_coded)
        final_seqs = np.array(final_tmp_seqs)
        final_seqs = final_seqs.flatten()
        
        final_seqs = np.reshape(final_seqs, (-1, max_len*len(self.coding)))
        print('the shape of final_seqs is', final_seqs.shape)
        
        return final_seqs, max_len, len(self.coding)
 

           
    