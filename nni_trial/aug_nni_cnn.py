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

import argparse
import logging

import os
#import keras
import numpy as np
#from keras import backend as K
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,BatchNormalization,Dropout
from keras.models import Sequential
from sklearn.metrics import roc_auc_score
import keras.optimizers as op
import keras.losses as losses
from keras.callbacks import Callback
import nni

import data_read


LOG = logging.getLogger('ayf_nni_cnn')
#K.set_image_data_format('channels_last')
TENSORBOARD_DIR = os.environ['NNI_OUTPUT_DIR']

H, W = 40, 40
CHANNELS=16
NUM_CLASSES = 2

def create_model(hyper_params, input_shape=(H, W, CHANNELS), num_classes=NUM_CLASSES):
    '''
    Create simple convolutional model
    '''
    
    layers = [
        BatchNormalization(axis=3, input_shape=input_shape),
        Conv2D(np.int32(hyper_params['Conv2D_1_number']), kernel_size=(np.int32(hyper_params['kernel_size']), np.int32(hyper_params['kernel_size'])), activation=hyper_params['activation']),
        Conv2D(np.int32(hyper_params['Conv2D_2_number']), kernel_size=(np.int32(hyper_params['kernel_size']),np.int32(hyper_params['kernel_size'])), activation=hyper_params['activation'],),
        MaxPooling2D(pool_size=(np.int32(hyper_params['pool_size']), np.int32(hyper_params['pool_size']))),
        Flatten(),
        Dense(np.int32(hyper_params['dense_size']), activation=hyper_params['activation'],),
        Dropout(hyper_params['Dropout_rate']),
        Dense(num_classes, activation='softmax')
    ]

    model = Sequential(layers)
    
    if hyper_params['optimizer'] == 'Adam':
        optimizer = op.Adam(lr=hyper_params['learning_rate'])
    else:
        optimizer = op.SGD(lr=hyper_params['learning_rate'], momentum=0.9)
    model.compile(loss=losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model


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
        LOG.debug(logs)
#        global test_x,test_y
        y_pred=self.model.predict_proba(self.x_val, verbose=0)
        score = roc_auc_score(self.y_val[:,1], y_pred[:,1])
        nni.report_intermediate_result(score)
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

def get_mixup_data(origion_x,origion_y,data_flip,data_rot,data_noise,params):
    data_aug_generator=data_read.data_aug(None,None)
    origion_y=data_read.label_to_onehot(origion_y)
    data_flip[1]=data_read.label_to_onehot(data_flip[1])
    data_rot[1]=data_read.label_to_onehot(data_rot[1])
    data_noise[1]=data_read.label_to_onehot(data_noise[1])
    if params['mixup_type']=='mixup':
        mixup_x=[]
        mixup_y=[]
        for i in range(params['mixup_k']):
            mixup_x1,mixup_y1=data_aug_generator.mixup(origion_x,origion_y,params['mixup_alpha'])
            mixup_x.append(mixup_x1)
            mixup_y.append(mixup_y1)
        mixup_x,mixup_y=data_aug_generator.merge_data(mixup_x,mixup_y)
        new_data0,new_data1=data_aug_generator.merge_data([mixup_x,origion_x],
                                                          [mixup_y,origion_y])
        return [new_data0,new_data1]
    elif params['mixup_type']=='basic_aug+mixup':
        aug_x,aug_y=get_aug_data(data_flip,data_rot,data_noise,params['aug_type'])
        mixup_x,mixup_y=data_aug_generator.mixup(aug_x,aug_y,params['mixup_alpha'])
        return [mixup_x,mixup_y]
def train(args, params):
    '''
    Train model
    '''
    data=data_read.yushan_data()
    train_x,train_y,test_x,test_y=data.get_train_data(tr_path='../data/yushan/yushan_tr_index.npy',
                                                      tt_path='../data/yushan/yushan_tt_index.npy',
                                                      data_type='3D')
#    data_aug_generator=data_read.data_aug(train_x,train_y)
#    rot_x,rot_y=data_aug_generator.rotate(train_x,train_y)
#    flip_x,flip_y=data_aug_generator.flip(train_x,train_y)
#    noise_x,noise_y=data_aug_generator.add_noise(train_x,train_y,sigma=0.5)
#    aug_data=get_mixup_data(train_x,train_y,[flip_x,flip_y],[rot_x,rot_y],[noise_x,noise_y],params)
#    train_x,train_y=aug_data
    train_y=data_read.label_to_onehot(train_y)
    test_y=data_read.label_to_onehot(test_y)
    model = create_model(params)
    SendMetric=SendMetrics(validation_data=(test_x,test_y))
    model.fit(train_x, train_y, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
        validation_data=(test_x, test_y), callbacks=[SendMetric, TensorBoard(log_dir=TENSORBOARD_DIR)])
    y_pred=model.predict_proba(test_x)
    score = roc_auc_score(test_y[:,1], y_pred[:,1])
#    _, acc = model.evaluate(x_test, y_test, verbose=0)
    LOG.debug('Final result is: %d', score)
    nni.report_final_result(score)

def generate_default_params():
    '''
    Generate default hyper parameters
    '''
    return {
        'optimizer': 'Adam',
        'learning_rate': 0.001
    }

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--batch_size", type=int, default=32, help="batch size", required=False)
    PARSER.add_argument("--epochs", type=int, default=50, help="Train epochs", required=False)
#    PARSER.add_argument("--num_train", type=int, default=60000, help="Number of train samples to be used, maximum 60000", required=False)
#    PARSER.add_argument("--num_test", type=int, default=10000, help="Number of test samples to be used, maximum 10000", required=False)

    ARGS, UNKNOWN = PARSER.parse_known_args()

    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = generate_default_params()
        PARAMS.update(RECEIVED_PARAMS)
        # train
        train(ARGS, PARAMS)
    except Exception as e:
        LOG.exception(e)
        raise
