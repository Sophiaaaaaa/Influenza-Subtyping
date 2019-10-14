from keras.models import load_model
from utils import *
from sklearn.metrics import accuracy_score
import numpy as np
from Resnet import Resnet
from sklearn.metrics import classification_report 

np.set_printoptions(threshold=np.inf)        


def max_class(scores, classes):
    
    max_p = 0
    for i in range(len(scores)):
        if scores[i]>max_p:
            max_p = scores[i]
    for i in range(len(scores)):
        if scores[i]==max_p:
            flag = i
            break
    return classes[flag]
        
class test():
    def __init__(self, args):
        self.type_name = args.type_name
        self.args = args
        # self.multi = args.multi
        
    def run(self):
        '''
        if self.multi==1:
            sequences, label, max_len, num_acid = load_multidata(self.type_name, isTrain=False)
            seq_test = sequences.reshape(sequences.shape[0], max_len, num_acid, 1)
            print('the shape of test seqs is', seq_test.shape)
            print('label is', label)
        
            history = LossHistory()
            res = Resnet('test', history, self.args)
            model = res.set_multimodel(max_len, num_acid)
            
            model.load_weights('./result/'+self.type_name+'/weight/multi_model_weights_8:2.h5')
            result = model.predict(seq_test, batch_size=64, verbose=0)
            predict = [-1 for i in range(len(result))]
            print('result is:', result)
            for i in range(len(result)):
                predict[i] = np.argmax(result[i])
            print('predict is:', predict)
            # evaluate
            accuracy = accuracy_score(label, predict)
            print('accuracy of test is', accuracy)
            print('class accuracy is:\n',classification_report(label, predict))
        '''    
            
        result_final = []
        predict = []
        if self.type_name=='HA':
            self.label_list = ['H1', 'H10', 'H11', 'H13', 'H16', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H9']
        elif self.type_name=='NA':
            self.label_list = ['N1', 'N2', 'N3', 'N5', 'N6', 'N7', 'N8', 'N9']
    
        label_dic = labels_dic(self.label_list)
    
    
        sequences, label, max_len, num_acid = load_dataset('test', self.type_name, isTrain=False)
        seq_test = sequences.reshape(sequences.shape[0], max_len, num_acid, 1)
        print('the shape of test seqs is', seq_test.shape)
        print('label is', label)
        print('set label is', set(label))
    
        history = LossHistory()
        res = Resnet('test', history, self.args)
        model = res.set_model(max_len, num_acid)
    
        # for each subtype, calculate voting score
        for l_name in self.label_list:
            print('current l_name is', l_name)
                  
            model.load_weights('./result/'+self.type_name+'/weight/model_weights_'+l_name+'_8:2.h5')
            result = model.predict(seq_test, batch_size=64, verbose=0)
            result_final.append(result)
            # print('result is', result)
            # print('the shape of result is', result.shape)
        result_final = np.array(result_final)
        print('final result shape is', result_final.shape)
        # print('final result is', result_final)
    
        # for each test example, cal the predict label
        for i in range(result_final.shape[1]):
            classes = []
            score = []
            # for each classifer, cal the possibility
            for j in range(len(self.label_list)):
                if result_final[j][i]>0.5:
                    classes.append(j)
                    score.append(result_final[j][i])
                    # print('score is', score)
                    # print('classes shape is', len(classes))
            if len(classes)==0:
                classes.append(-1)
            if len(classes)==1:
                predict.append(classes[0])
            else:
                predict.append(max_class(score, classes))
            
        print('predict is', predict)
        predict = np.array(predict)
        print('set predict is', set(predict))
    
        # evaluate
        accuracy = accuracy_score(label, predict)
        print('accuracy of test is', accuracy)
        print('class accuracy is:\n',classification_report(label, predict))