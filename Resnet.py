import keras
from keras.layers import merge, Lambda, Reshape, concatenate
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding1D, AveragePooling2D, Conv1D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.datasets import mnist
from keras import Input
from scipy import interp

import numpy as np
from utils import batch_size

class Resnet():
    def __init__(self, l_name, history, args):
        self.batch_size = batch_size(l_name)
        self.nb_epoch = args.epoch
        self.lr = args.lr
        self.type_name = args.type_name
        # self.multi = args.multi
        self.nb_filter_1 = 256
        self.filter_size_1 = 3
        self.strides_1 = 1
        self.class_num = 1
        self.l_name = l_name
        self.history = history
        '''
        if args.type_name=='HA':
            self.mulclass_num = 12
        elif args.type_name=='NA':
            self.mulclass_num = 8
        '''
         
    # concatenate when the same num_filter
    def identify_block(self, x,nb_filter,kernel_size = 3):
        # k1, k2, k3 = nb_filter
        k1, k2 = nb_filter
        out = Convolution2D(k1, (kernel_size, kernel_size), padding='same')(x)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Convolution2D(k2, (kernel_size, kernel_size), padding='same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = concatenate([out, x],axis=-1)
        out = Activation('relu')(out)
        return out
    
    # concatenate when different num_filter
    def conv_block(self, x, nb_filter, kernel_size = 3):
        # k1, k2, k3 = nb_filter
        k1, k2 = nb_filter
        out = Convolution2D(k1,(kernel_size, kernel_size), padding = 'same')(x)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Convolution2D(k2, (kernel_size,kernel_size), padding = 'same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        x = Convolution2D(k2, (1, 1), padding = 'valid')(x)
        x = BatchNormalization()(x)

        out = concatenate([out,x],axis=-1)
        out = Activation('relu')(out)
        return out    

    def set_model(self, rows, cols):
    
        inp = Input(shape=(rows, cols, 1))
        
        # an 256*cols kernel 
        out = Convolution2D(self.nb_filter_1, (self.filter_size_1, cols), strides=self.strides_1)(inp)
        out = Activation('relu')(out)
        out = Reshape((int((rows-self.filter_size_1) / self.strides_1) + 1, self.nb_filter_1, 1))(out)
        out = AveragePooling2D((3, 3))(out)
        
        out = self.conv_block(out, [16, 16])
        out = self.identify_block(out, [16, 16])
        out = self.identify_block(out, [16, 16])
        out = self.identify_block(out, [16, 16])
        out = AveragePooling2D((3, 3))(out)
        
        out = self.conv_block(out, [32, 32])
        out = self.identify_block(out, [32, 32])
        out = self.identify_block(out, [32, 32])
        out = self.identify_block(out, [32, 32])
        out = AveragePooling2D((3, 3))(out)
        
        
        out = self.conv_block(out, [64, 64])
        out = self.identify_block(out, [64, 64])
        out = self.identify_block(out, [64, 64])
        out = self.identify_block(out, [64, 64])       
        # out = AveragePooling2D((3, 3))(out)
    
        out = Flatten()(out)
        out = Dense(64, activation='relu', name='Dense_1')(out)
        out = Dropout(0.5)(out)
        out = Dense(self.class_num, activation='sigmoid', name='Dense_2')(out)

        model = Model(inp, out)
        model.summary()
        # plot_model(model, to_file='model.png', show_shapes=True)
        # compile a model
        adam = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # sgd =SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=adam)
        return model
    
    def set_multimodel(self, rows, cols):
    
        inp = Input(shape=(rows, cols, 1))
        
        # an 256*cols kernel 
        out = Convolution2D(self.nb_filter_1, (self.filter_size_1, cols), strides=self.strides_1)(inp)
        out = Activation('relu')(out)
        out = Reshape((int((rows-self.filter_size_1) / self.strides_1) + 1, self.nb_filter_1, 1))(out)
        out = AveragePooling2D((3, 3))(out)
        
        out = self.conv_block(out, [16, 16])
        out = self.identify_block(out, [16, 16])
        out = self.identify_block(out, [16, 16])
        out = self.identify_block(out, [16, 16])
        out = AveragePooling2D((3, 3))(out)
        
        out = self.conv_block(out, [32, 32])
        out = self.identify_block(out, [32, 32])
        out = self.identify_block(out, [32, 32])
        out = self.identify_block(out, [32, 32])
        out = AveragePooling2D((3, 3))(out)
        
        
        out = self.conv_block(out, [64, 64])
        out = self.identify_block(out, [64, 64])
        out = self.identify_block(out, [64, 64])
        out = self.identify_block(out, [64, 64])     
        # out = AveragePooling2D((3, 3))(out)
    
        out = Flatten()(out)
        out = Dense(64, activation='relu', name='Dense_1')(out)
        out = Dropout(0.5)(out)
        out = Dense(self.mulclass_num, activation='sigmoid', name='Dense_2')(out)

        model = Model(inp, out)
        model.summary()
        # plot_model(model, to_file='model.png', show_shapes=True)
        # compile a model
        adam = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # sgd =SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=adam)
        return model
    
    def run(self, model, x_train, x_val, y_train, y_val):
        base_path = './result/'+self.type_name+'/weight/'
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='min', baseline=None, restore_best_weights=False)
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, mode='min', min_delta=0.000001)
        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.nb_epoch, validation_data=(x_val, y_val), verbose=1,
              shuffle=True, callbacks=[self.history, early_stop])
        # if self.multi==0:
        model.save_weights(base_path+'model_weights_'+self.l_name+'_8:2.h5', overwrite=True)
        # else:
            # model.save_weights(base_path+'multi_model_weights_8:2.h5', overwrite=True)