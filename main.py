import argparse
from utils import *
from data_processing import prepareData
from data_set import data_set
from train import train
from test import test

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

'''
# set gpu config
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
'''
np.set_printoptions(threshold=np.inf)

'''parsing and configuration'''
def parse_args():
    desc = "Tensorflow implementation of ResNet"
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument('--phase', type=str, default='train', help='train or test?')
    parser.add_argument('--type_name', type=str, default='HA', help='[HA, NA]')
    parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    # parser.add_argument('--multi', type=int, default=0, help='yes 1, no 0')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    
    return check_args(parser.parse_args())
    
'''checking arguments'''
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)
    
    # --result_dir
    check_folder(args.log_dir)
    
    # --epoch
    try:
        assert args.epoch>=1
    except:
        print('number of epochs must be larger than or equal to one')
        
    return args
    
'''main'''
def main():
    dataset_path = './data_set/'
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
        
    # make data set if preprocessing data not exist
    
    if not os.listdir(dataset_path+args.type_name):
        # select sequences after year 2010
        prepare = prepareData(args.type_name)
        ds_path = prepare.select()
        # split train and test data
        data = data_set(args.type_name)
        data.form_dataset()
    elif os.path.exists(dataset_path+args.type_name+'/data_label_'+args.type_name+'_IVD.npz') and not os.path.exists(dataset_path+args.type_name+'/test_'+args.type_name+'_8:2.npz'):
        print('skip preprocessing!')
        data = data_set(args.type_name)
        data.form_dataset()
    
    
    
    if args.phase == 'train':
        tr = train(args)
        tr.run()
        
    elif args.phase == 'test':
        te = test(args)
        te.run()
        
        
if __name__=='__main__':
    main()
    
    