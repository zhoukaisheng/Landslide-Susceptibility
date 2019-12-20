# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:13:23 2019

@author: 75129
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import  OneHotEncoder


BASE_DIR=os.path.dirname(__file__)
BASE_DIR_a_c='D:/yanshan_new/'
class from_a_c_data():
    def __init__(self,a,c):
        self.x_3d=np.load(os.path.join(BASE_DIR_a_c,'resize/a'+str(a)+'_c'+str(int(100*c)).zfill(3)+'.npy'))
        self.x_1d=pd.read_csv(os.path.join(BASE_DIR_a_c,'origion_shape/a'+str(a)+'_c'+str(int(100*c)).zfill(3)+'/yinzi1d.csv'))
        self.x_1d=np.array(self.x_1d)
        self.x_1d=self.x_1d[:,1:]
        self.label=np.load(os.path.join(BASE_DIR_a_c,'tr_tt_npy_and_label/label_a'+str(a)+'_c'+str(int(100*c)).zfill(3)+'.npy'))
        self.label=self.label.reshape([self.label.shape[0],])
        self.a=a
        self.c=c
        # self.label=self.label['landslide']
    def get_train_data(self,data_type='3D',onehot=False):
        a=self.a
        c=self.c
        tr_path=os.path.join(BASE_DIR_a_c,'tr_tt_npy_and_label/tr_a'+str(a)+'_c'+str(int(100*c)).zfill(3)+'.npy')
        tt_path=os.path.join(BASE_DIR_a_c,'tr_tt_npy_and_label/tt_a'+str(a)+'_c'+str(int(100*c)).zfill(3)+'.npy')
        train_index=np.load(tr_path)
        train_index=train_index.reshape([train_index.shape[1],])
        test_index=np.load(tt_path)
        test_index=test_index.reshape([test_index.shape[1],])
        # if tr_path==None and tt_path==None:
        #     train_index,test_index=TrainIndexSelect(train_rate,self.label)
        # else:
        #     train_index=np.load(tr_path)
        #     test_index=np.load(tt_path)
        # if onehot==True:
        #     str_list=[13,14,15]
        #     tmp=to_onehot(self.x_1d[:,str_list])
        #     self.x_1d=np.delete(self.x_1d,str_list,axis=1)
        #     self.x_1d=np.column_stack((self.x_1d,tmp))
        if data_type=='3D':
            all_x=self.x_3d
            train_x=all_x[train_index,:,:,:]
            test_x=all_x[test_index,:,:,:]
        elif data_type=='1D':
            all_x=self.x_1d
            train_x=all_x[train_index,:]
            test_x=all_x[test_index,:]
        train_y=self.label[train_index]
        test_y=self.label[test_index]

        return train_x,train_y,test_x,test_y


class yushan_data():
    def __init__(self):
        self.x_3d=np.load(os.path.join(BASE_DIR,'data/yushan/all_yushan.npy'))
        self.x_1d=pd.read_csv(os.path.join(BASE_DIR,'data/yushan/yinzi1d.csv'))
        self.x_1d=np.array(self.x_1d)
        self.x_1d=self.x_1d[:,1:]
        self.label=pd.read_csv(os.path.join(BASE_DIR,'data/yushan/yushan.csv'))
        self.label=self.label['landslide']
    def get_train_data(self,tr_path=None,tt_path=None,train_rate=0.7,data_type='3D',onehot=False):
        if tr_path==None and tt_path==None:
            train_index,test_index=TrainIndexSelect(train_rate,self.label)
        else:
            train_index=np.load(tr_path)
            test_index=np.load(tt_path)
        if onehot==True:
            str_list=[13,14,15]
            tmp=to_onehot(self.x_1d[:,str_list])
            self.x_1d=np.delete(self.x_1d,str_list,axis=1)
            self.x_1d=np.column_stack((self.x_1d,tmp))
        if data_type=='3D':
            all_x=self.x_3d
            train_x=all_x[train_index,:,:,:]
            test_x=all_x[test_index,:,:,:]
        elif data_type=='1D':
            all_x=self.x_1d
            train_x=all_x[train_index,:]
            test_x=all_x[test_index,:]
        train_y=self.label[train_index]
        test_y=self.label[test_index]

        return train_x,train_y,test_x,test_y

class yongxin_data():
    def __init__(self):
        self.x_3d=np.load(os.path.join(BASE_DIR,'data/yongxin/all_yongxin.npy'))
        self.x_3d_bno0=np.load(os.path.join(BASE_DIR,'data/yongxin/all_yongxin_NO0.npy'))
        self.x_1d=pd.read_csv(os.path.join(BASE_DIR,'data/yongxin/yinzi1d.csv'))
        self.x_1d=np.array(self.x_1d)
        self.x_1d=self.x_1d[:,1:]
        self.label=pd.read_csv(os.path.join(BASE_DIR,'data/yongxin/yongxin.csv'))
        self.label=self.label['landslide']
    def get_train_data(self,tr_path=None,tt_path=None,train_rate=0.7,data_type='3D',onehot=False,no0=False):
        if no0==False:
            x_3d=self.x_3d
        elif no0==True:
            x_3d=self.x_3d_bno0
        if tr_path==None and tt_path==None:
            train_index,test_index=TrainIndexSelect(train_rate,self.label)
        else:
            train_index=np.load(tr_path)
            test_index=np.load(tt_path)
        if onehot==True:
            str_list=[13,14,15]
            tmp=to_onehot(self.x_1d[:,str_list])
            self.x_1d=np.delete(self.x_1d,str_list,axis=1)
            self.x_1d=np.column_stack((self.x_1d,tmp))
        if data_type=='3D':
            all_x=x_3d
            train_x=all_x[train_index,:,:,:]
            test_x=all_x[test_index,:,:,:]
        elif data_type=='1D':
            all_x=self.x_1d
            train_x=all_x[train_index,:]
            test_x=all_x[test_index,:]
        train_y=self.label[train_index]
        test_y=self.label[test_index]

        return train_x,train_y,test_x,test_y


        
        

        
def TrainIndexSelect(trainRate,alllable):
    SlideIndex=np.where(alllable>0)[0]
    NoSlideIndex=np.where(alllable==0)[0]
    slidenum=SlideIndex.shape[0]
    train_slnum=int(slidenum*trainRate)
    slidenum_index=np.random.permutation(slidenum)
    train_index_sl=SlideIndex[slidenum_index[:train_slnum]]
    test_index_sl=SlideIndex[slidenum_index[train_slnum:]]
    noSlidenum_index=np.random.permutation(NoSlideIndex.shape[0])
    train_index_nosl=NoSlideIndex[noSlidenum_index[:train_slnum]]
    test_index_nosl=NoSlideIndex[noSlidenum_index[train_slnum:(train_slnum+(slidenum-train_slnum))]]
    train_index=np.append(train_index_nosl,train_index_sl)
    test_index=np.append(test_index_nosl,test_index_sl)
    return train_index,test_index

def batch_generator(all_data , batch_size, shuffle=True):
    all_data = [np.array(d) for d in all_data]
    data_size = all_data[0].shape[0]
    print("data_size: ", data_size)
    if shuffle:
        p = np.random.permutation(data_size)
        all_data = [d[p] for d in all_data]

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > data_size:
            batch_count = 0
            if shuffle:
                p = np.random.permutation(data_size)
                all_data = [d[p] for d in all_data]
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start: end] for d in all_data]
        
class data_aug():
    def __init__(self,data_x,data_y):
        self.data_x=data_x
        self.data_y=data_y
    def widen(self,widen_size=1):
        shape=np.shape(self.data_x)
        widen_x=np.zeros([shape[0],shape[1]+widen_size*2,shape[2]+widen_size*2,shape[3]])
        widen_x[:,widen_size:-widen_size,widen_size:-widen_size,:]=self.data_x
        widen_y=self.data_y
        return widen_x,widen_y
    def clip(self,data_x,data_y,shape=[41,41,16],edge_width=1,):
        if edge_width!=1:
            print('Only support edge_width = 1 !')
        left_up=data_x[:,:-1,:-1,:]
        left_down=data_x[:,:-1,1:,:]
        right_up=data_x[:,1:,:-1,:]
        right_down=data_x[:,1:,1:,:]
        clip_x,clip_y=self.merge_data([left_up,left_down,right_up,right_down],[data_y,data_y,data_y,data_y])
        return clip_x,clip_y
    def rotate(self,data_x,data_y):
        rot_data_x=[]
        rot_data_x.append(np.rot90(data_x,k=1,axes=(1,2)))
        rot_data_x.append(np.rot90(data_x,k=2,axes=(1,2)))
        rot_data_x.append(np.rot90(data_x,k=3,axes=(1,2)))
        return self.merge_data(rot_data_x,[data_y,data_y,data_y])
    def flip(self,data_x,data_y):
        flip_data_x=[]
        flip_data_x.append(np.flip(data_x,axis=1))
        flip_data_x.append(np.flip(data_x,axis=2))
        return self.merge_data(flip_data_x,[data_y,data_y])
    
    def merge_data(self,data_x_list,data_y_list):
#        lenth=len(data_x_list)
        merge_x=np.concatenate(data_x_list,axis=0)
        merge_y=np.concatenate(data_y_list,axis=0)
        return merge_x,merge_y
    
    def mixup(self,x, y, alpha=0.2):
        candidates_data, candidates_label = x, y
#        offset = (step * batch_size) % (candidates_data.shape[0] - batch_size)
#        train_features_batch = candidates_data[offset:(offset + batch_size)]
#        train_labels_batch = candidates_label[offset:(offset + batch_size)]
        train_features_batch=x
        train_labels_batch=y
        shape=np.shape(train_features_batch)
        if alpha == 0:
            return train_features_batch, train_labels_batch
        if alpha > 0:
            weight = np.random.beta(alpha, alpha, shape[0])
            x_weight = weight.reshape(shape[0], 1, 1, 1)
            y_weight = weight.reshape(shape[0], 1)
            index = np.random.permutation(shape[0])
            x1, x2 = train_features_batch, train_features_batch[index]
            x = x1 * x_weight + x2 * (1 - x_weight)
            y1, y2 = train_labels_batch, train_labels_batch[index]
            y = y1 * y_weight + y2 * (1 - y_weight)
            return x, y
    def sample_pairing(self,x, y):
        candidates_data, candidates_label = x, y
#        offset = (step * batch_size) % (candidates_data.shape[0] - batch_size)
#        train_features_batch = candidates_data[offset:(offset + batch_size)]
#        train_labels_batch = candidates_label[offset:(offset + batch_size)]
        train_features_batch=x
        train_labels_batch=y
        shape=np.shape(train_features_batch)
        index = np.random.permutation(shape[0])
        x1, x2 = train_features_batch, train_features_batch[index]
        x = x1 * 0.5 + x2 * 0.5
        y1, y2 = train_labels_batch, train_labels_batch[index]
        y = y1
        return x, y
    def add_noise(self,data_x,data_y,sigma):
        noise=np.random.normal(0,sigma,data_x.shape)
        noise_x=data_x+noise
        return noise_x,data_y
        
        
        
        
def label_to_onehot(label):
    "Only two categories are supported"
    onehot=np.zeros([label.shape[0],2])
    onehot[:,0]=1-label
    onehot[:,1]=label
    return onehot

def to_onehot(all_lable):
    enc=OneHotEncoder()
    all_lable=all_lable+1
    enc.fit(all_lable)

    all_lable = enc.transform(all_lable).toarray()
    shape=all_lable.shape
    zero=np.zeros([shape[0],1])
    zero_index=[]
    for i in range(shape[1]):
        tmp=all_lable[i,:]==zero
        if tmp.all():
            zero_index.append(i)
    all_lable=np.delete(all_lable,zero_index,axis=1)
    return all_lable


    
            
if __name__ == '__main__':
    data=from_a_c_data(a=10000,c=0.1)
    train_x,train_y,test_x,test_y=data.get_train_data(data_type='1D')
    # train_index,test_index=TrainIndexSelect(0.7,data.label)
    # np.save('tr_index',train_index)
    # np.save('tt_index',test_index)
#     batch_gen=batch_generator([train_x,train_y] , batch_size=32, shuffle=True)
# #    for i in range(5):
# #        image_batch, label_batch=next(batch_gen)
# #        print(image_batch[0,15,20,7])
#     data_au=data_aug(train_x,train_y)
#     widen_x,widen_y=data_au.widen(widen_size=1)
#     train_y_onehot=label_to_onehot(train_y)
#     new_x,new_y=data_au.add_noise(test_x, test_y,10)
    print(data.x_1d.shape)
    print(data.x_3d.shape)
    print(data.label.shape)
    print(train_x.shape)
    print(test_x.shape)
