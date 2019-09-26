# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 20:18:22 2019

@author: 75129
"""

# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import os
import argparse
#import keras
import numpy as np
#from keras import backend as K
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,BatchNormalization,Dropout,Input,merge,SeparableConv2D,DepthwiseConv2D,Conv3D,MaxPooling3D
from keras.models import Sequential,Model
from sklearn.metrics import roc_auc_score
import keras.optimizers as op
import keras.losses as losses
from keras.callbacks import Callback
from keras.backend import concatenate
import nni
import sys
sys.path.append('D:/myGIT/Landslide-Susceptibility')
import data_read
import pandas as pd

# np.random.seed(33)  /3d
np.random.seed(27)

#K.set_image_data_format('channels_last')

H, W = 40, 40
CHANNELS=16
NUM_CLASSES = 2


def create_3d_model(inputs,hyper_params):
    # inputs=Input(shape=(40,40,16))
    x=BatchNormalization()(inputs)
    x=DepthwiseConv2D(kernel_size=(3,3),activation='relu')(x)
    x=MaxPooling2D(pool_size=(3,3))(x)
    x=DepthwiseConv2D(kernel_size=(3,3),activation='relu')(x)
    x=MaxPooling2D(pool_size=(3,3))(x)
    x=DepthwiseConv2D(kernel_size=(3,3),activation='relu')(x)
    x=Flatten()(x)
    return x

def create_mixture_model(hyper_params):
    inputs_3d=Input(shape=(40,40,16))
    x_3d=create_3d_model(inputs_3d,hyper_params)
    x_1d=Input(shape=(4,))
    mixtured=merge.concatenate([x_3d,x_1d])
    z=BatchNormalization()(mixtured)
    z=Dense(np.int32(hyper_params['dense_size']),activation='relu')(z)
    z=Dropout(hyper_params['Dropout_rate'])(z)
    z=Dense(2,activation='softmax')(z)
    model=Model(inputs=[inputs_3d,x_1d],outputs=z)
    if hyper_params['optimizer'] == 'Adam':
        optimizer = op.Adam(lr=hyper_params['learning_rate'])
    else:
        optimizer = op.SGD(lr=hyper_params['learning_rate'], momentum=0.9)
    model.compile(loss=losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

def create_Only3D_model(hyper_params):
    inputs_3d=Input(shape=(40,40,16))
    x_3d=create_3d_model(inputs_3d,hyper_params)
    # x_1d=Input(shape=(20,))
    # mixtured=merge.concatenate([x_3d,x_1d])
    z=BatchNormalization()(x_3d)
    z=Dense(np.int32(hyper_params['dense_size']),activation='relu')(z)
    z=Dropout(hyper_params['Dropout_rate'])(z)
    z=Dense(2,activation='softmax')(z)
    model=Model(inputs=inputs_3d,outputs=z)
    if hyper_params['optimizer'] == 'Adam':
        optimizer = op.Adam(lr=hyper_params['learning_rate'])
    else:
        optimizer = op.SGD(lr=hyper_params['learning_rate'], momentum=0.9)
    model.compile(loss=losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

def create_Only1D_model(hyper_params):
    # inputs_3d=Input(shape=(40,40,16))
    # x_3d=create_3d_model(inputs_3d,hyper_params)
    x_1d=Input(shape=(16,))
    # mixtured=merge.concatenate([x_3d,x_1d])
    z=BatchNormalization()(x_1d)
    z=Dense(np.int32(hyper_params['dense_size']),activation='relu')(z)
    z=Dropout(hyper_params['Dropout_rate'])(z)
    z=Dense(2,activation='softmax')(z)
    model=Model(inputs=x_1d,outputs=z)
    if hyper_params['optimizer'] == 'Adam':
        optimizer = op.Adam(lr=hyper_params['learning_rate'])
    else:
        optimizer = op.SGD(lr=hyper_params['learning_rate'], momentum=0.9)
    model.compile(loss=losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

def get_intermediate_output(model,output_layer,input_data):
    model_new=Model(inputs=model.input,outputs=model.get_layer(output_layer).output)
    output=model_new.predict(input_data)
    return output
    

class SendMetrics(Callback):
    '''
    Keras callback to send metrics to NNI framework
    '''
    def __init__(self, validation_data=()):
        super(Callback, self).__init__()
        self.x_val,self.y_val = validation_data
    def on_epoch_end(self, epoch, logs={}):
        '''
        Run on end of each epoch
        '''
#        global test_x,test_y
        y_pred=self.model.predict(self.x_val, verbose=0)
        score = roc_auc_score(self.y_val[:,1], y_pred[:,1])
        print(score)


def get_aug_data(data_flip,data_rot,data_noise,aug_type):
    data_aug_generator=data_read.data_aug(None,None)
    if aug_type=="flip":
        return data_flip
    elif aug_type=="rot":
        return data_rot
    elif aug_type=="noise":
        return data_noise
    elif aug_type=="flip and rot":
        new_data0,new_data1=data_aug_generator.merge_data([data_flip[0],data_rot[0]],
                                                          [data_flip[1],data_rot[1]])
        return [new_data0,new_data1]
    elif aug_type=="flip and noise":
        new_data0,new_data1=data_aug_generator.merge_data([data_flip[0],data_noise[0]],
                                                          [data_flip[1],data_noise[1]])
        return [new_data0,new_data1]
    elif aug_type=="rot and noise":
        new_data0,new_data1=data_aug_generator.merge_data([data_rot[0],data_noise[0]],
                                                          [data_rot[1],data_noise[1]])
        return [new_data0,new_data1]
    elif aug_type=="all":
        new_data0,new_data1=data_aug_generator.merge_data([data_rot[0],data_noise[0],data_flip[0]],
                                                          [data_rot[1],data_noise[1],data_flip[1]])
        return [new_data0,new_data1]

def train(args, params):
    '''
    Train model
    '''
    data=data_read.yongxin_data()
    train_x3d,train_y,test_x3d,test_y=data.get_train_data(tr_path='D:/myGIT/Landslide-Susceptibility/data/yongxin/tr_index.npy',
                                                      tt_path='D:/myGIT/Landslide-Susceptibility./data/yongxin/tt_index.npy',
                                                      data_type='3D')
    train_x1d,train_y,test_x1d,test_y=data.get_train_data(tr_path='D:/myGIT/Landslide-Susceptibility/data/yongxin/tr_index.npy',
                                                      tt_path='D:/myGIT/Landslide-Susceptibility/data/yongxin/tt_index.npy',
                                                      data_type='1D')                                    
#    data_aug_generator=data_read.data_aug(train_x,train_y)
#    rot_x,rot_y=data_aug_generator.rotate(train_x,train_y)
#    flip_x,flip_y=data_aug_generator.flip(train_x,train_y)
#    noise_x,noise_y=data_aug_generator.add_noise(train_x,train_y,sigma=0.5)
#    aug_data=get_mixup_data(train_x,train_y,[flip_x,flip_y],[rot_x,rot_y],[noise_x,noise_y],params)
#    train_x,train_y=aug_data
    train_y=data_read.label_to_onehot(train_y)
    test_y=data_read.label_to_onehot(test_y)
    model = create_mixture_model(params)
    # train_x3d=np.expand_dims(train_x3d,axis=5)
    # test_x3d=np.expand_dims(test_x3d,axis=5)
    train_data=[train_x3d,train_x1d[:,-4:]]
    test_data=[test_x3d,test_x1d[:,-4:]]
    SendMetric=SendMetrics(validation_data=(test_data,test_y))
    model.fit(train_data, train_y,  epochs=args.epochs, verbose=1,
        validation_data=(test_data, test_y), callbacks=[SendMetric])
    y_pred=model.predict(test_data)
    score = roc_auc_score(test_y[:,1], y_pred[:,1])
#    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Final result is: %d', score)
    all_pro=model.predict([data.x_3d,data.x_1d[:,-4:]])
    df = pd.DataFrame(all_pro)
    df.to_csv('C:/Users/75129/Desktop/nni实验记录/proba_mixed.csv')
    extract_yinzi=get_intermediate_output(model,'flatten_1',[data.x_3d,data.x_1d[:,-4:]])
    df = pd.DataFrame(extract_yinzi)
    df.to_csv('C:/Users/75129/Desktop/nni实验记录/cnn_extract_yinzi.csv')
    

def generate_default_params():
    '''
    Generate default hyper parameters
    '''



    parameters={"optimizer":"Adam","learning_rate":0.00025690063520965767,"Conv2D_1_number":99,"Conv2D_2_number":13,"dense_size":32,"Dropout_rate":0.7329247335728416}

    return parameters

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--batch_size", type=int, default=32, help="batch size", required=False)
    PARSER.add_argument("--epochs", type=int, default=100, help="Train epochs", required=False)
#    PARSER.add_argument("--num_train", type=int, default=60000, help="Number of train samples to be used, maximum 60000", required=False)
#    PARSER.add_argument("--num_test", type=int, default=10000, help="Number of test samples to be used, maximum 10000", required=False)

    ARGS, UNKNOWN = PARSER.parse_known_args()

    try:
        # get parameters from tuner
        PARAMS = generate_default_params()
        # train
        train(ARGS, PARAMS)
    except Exception as e:
        raise
