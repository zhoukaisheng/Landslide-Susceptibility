# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 20:50:19 2019

@author: 75129
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 20:18:22 2019

@author: 75129
"""



import argparse

import os
#import keras
import numpy as np
#from keras import backend as K
from sklearn.metrics import roc_auc_score,accuracy_score
import nni
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append('D:\myGIT\Landslide-Susceptibility')
import data_read

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
#K.set_image_data_format('channels_last')
import pandas as pd

NUM_CLASSES = 2



def train_svm(args):
    '''
    Train model
    '''
    data=data_read.yushan_data()
    train_x,train_y,test_x,test_y=data.get_train_data(tr_path='D:/myGIT/Landslide-Susceptibility/data/yushan/yushan_tr_index.npy',
                                                      tt_path='D:/myGIT/Landslide-Susceptibility/data/yushan/yushan_tt_index.npy',
                                                      data_type='1D',onehot=True)
    parameters={'C':[1/32,1/16,1/8,1/4,1/2,1,2,4,8,16,32],'gamma':[1/32,1/16,1/8,1/4,1/2,1,2,4,8,16,32]}
    x_1d=data.x_1d
    
    mm = MinMaxScaler()
    mm.fit(x_1d)
    train_x=mm.transform(train_x)
    test_x=mm.transform(test_x)
    shape=train_x.shape
    x_1d=mm.transform(x_1d)                                                                                                                  
    # train_y=data_read.label_to_onehot(train_y)
    # test_y=data_read.label_to_onehot(test_y)
    clf=svm.SVC(probability = True,kernel='rbf')
    clf=GridSearchCV(clf,parameters,cv=5,verbose=3)
    print('begin GridSearchCV')
    # print(train_y)
    clf.fit(train_x,train_y)
    y_pred=clf.predict_proba(test_x)
    score = roc_auc_score(test_y, y_pred[:,1])
    print('auc:'+str(score))
    acc=accuracy_score(test_y,y_pred[:,1]>0.5)
    print('acc:'+str(acc))
    print(train_x.shape)
    y_pred_all=clf.predict_proba(x_1d)
    df = pd.DataFrame(x_1d)
    df.to_csv('C:/Users/75129/Desktop/nni实验记录/proba_svm.csv')
#    _, acc = model.evaluate(x_test, y_test, verbose=0)

def train_knn(args):
    '''
    Train model
    '''
    data=data_read.yongxin_data()
    train_x,train_y,test_x,test_y=data.get_train_data(tr_path='D:/myGIT/Landslide-Susceptibility/data/yongxin/tr_index.npy',
                                                      tt_path='D:/myGIT/Landslide-Susceptibility/data/yongxin/tt_index.npy',
                                                      data_type='1D',onehot=False)
    parameters={'n_neighbors':[3,5,10,30,50,100],'algorithm':['auto','ball_tree','kd_tree','brute']}
    x_1d=data.x_1d
    
    mm = MinMaxScaler()
    mm.fit(x_1d)
    train_x=mm.transform(train_x)
    test_x=mm.transform(test_x)
    shape=train_x.shape
    x_1d=mm.transform(x_1d)                                                                                                                  
    # train_y=data_read.label_to_onehot(train_y)
    # test_y=data_read.label_to_onehot(test_y)
    clf=KNeighborsClassifier()
    clf=GridSearchCV(clf,parameters,cv=5,verbose=3)
    print('begin GridSearchCV')
    # print(train_y)
    clf.fit(train_x,train_y)
    y_pred=clf.predict_proba(test_x)
    score = roc_auc_score(test_y, y_pred[:,1])
    print('auc:'+str(score))
    acc=accuracy_score(test_y,y_pred[:,1]>0.5)
    print('acc:'+str(acc))
    print(train_x.shape)
    y_pred_all=clf.predict_proba(x_1d)
    df = pd.DataFrame(x_1d)
    df.to_csv('C:/Users/75129/Desktop/nni实验记录/proba_svm.csv')
    


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
#    PARSER.add_argument("--batch_size", type=int, default=32, help="batch size", required=False)
#    PARSER.add_argument("--epochs", type=int, default=50, help="Train epochs", required=False)
#    PARSER.add_argument("--num_train", type=int, default=60000, help="Number of train samples to be used, maximum 60000", required=False)
#    PARSER.add_argument("--num_test", type=int, default=10000, help="Number of test samples to be used, maximum 10000", required=False)

    ARGS, UNKNOWN = PARSER.parse_known_args()

    try:
        # get parameters from tuner
        # train
        train_knn(ARGS)
    except Exception as e:
        raise
