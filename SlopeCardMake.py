# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:13:10 2018

@author: 75129
"""

import scipy.io as sio
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat
from sklearn.decomposition import PCA
from scipy import stats
import pandas as pd
import imageio
import os
# import arcpy
# totaldata读取

def SUCardMake(su_path,a,c):
    yinzi_name = ['altitude', 'aspect', 'fault', 'ndvi', 'plan', 'profile', 'rainfall',
        'river', 'road', 'slope', 'spi', 'sti', 'twi', 'soil', 'landuse', 'lithology']
    yinzi_type_ori = ['float', 'float', 'float', 'float', 'float', 'float', 'float',
        'float', 'float', 'float', 'float', 'float', 'float', 'str', 'str', 'str']

    lable = imageio.imread(su_path)
    row = len(lable)
    col = len(lable[0])
    nums_factors = len(yinzi_name)
    img = np.zeros((row, col, nums_factors))
    i = 0
    yinzi_type = []
    for i in range(nums_factors):
        name = yinzi_name[i]
        PATH = 'C:/Users/75129/Desktop/mypy/demo_all/tif_yanshan/'
        img[:, :, i] = imageio.imread(PATH+name+'.tif')
    #    if yinzi_type_ori[i]=='str':
    #        plt.pyplot.figure()
    #        plt.pyplot.imshow(img[:,:,i])
        ttt = img[:, :, i]
        ttt[ttt == -32768] = 0
    #    ttt[ttt==-3.402823e+038]=0
        img[:, :, i] = ttt
    #    img_i=load_data[name]
    #    ma=np.max(img_i)
    #    mi=np.min(img_i)
    #    img[:,:,i]=(img_i-mi)/(ma-mi)
        i = i+1
    # pca = PCA(n_components=3)
    # img1=np.reshape(img,(2636*2206,16))
    # img_pca=pca.fit_transform(img1)
    # img_pca=np.reshape(img_pca,(2636,2206,3))
    # img=img_pca
    # plt.pyplot.imshow(img_pca[:,:,0])
    xfanwei = []
    yfanwei = []
    print('开始裁剪 。')
    empty_index = []
    max_id = np.max(lable[lable != 65535])
    yinzi_1d = np.zeros([max_id+1, nums_factors+2])
    for it in range(max_id+1):  # 4890
        fanwei = (lable == it)
        fanwei = fanwei+np.zeros((row, col))
        img_it = np.zeros((row, col, nums_factors+1))
        for i in range(nums_factors):
            img_it[:, :, i] = fanwei*img[:, :, i]
        # img_it[:,:,-1]=fanwei
        index_fanwei = np.argwhere(lable == it)
        if len(index_fanwei) != 0:
            x_min = min(index_fanwei[:, 0])
            x_max = max(index_fanwei[:, 0])
            y_min = min(index_fanwei[:, 1])
            y_max = max(index_fanwei[:, 1])
            img_it_resize = img_it[x_min:(x_max+1), y_min:(y_max+1), :]
            # img_it_resize = img[x_min:(x_max+1), y_min:(y_max+1), :]
            # img_it_resize=np.insert(img_it_resize,-1,values=img_it_resize1[:,:,-1],axis=2)
            # for i in range(nums_factors+1):
            #    plt.figure()
            #    plt.imshow(img_it_resize[:,:,i])
            #    plt.show()
            # print(img_it_resize.shape)
            filename='D:/yanshan_new/origion_shape/'+'a'+str(a)+'_c'+str(int(100*c)).zfill(3)+'/'+str(it)
            np.save(filename,img_it_resize)
            xfanwei.append(x_max-x_min+1)
            yfanwei.append(y_max-y_min+1)
            # img_it_resize=img_it_resize1
            for f in range(nums_factors):
                x,y,z=img_it_resize.shape
                values=np.reshape(img_it_resize,[x*y,z])
                values_f=values[:,f]
                values_f=values_f[values_f!=0]
                if yinzi_type_ori[f]=='str':
                    if values_f.size!=0:
                        yinzi_1d[it,f]=stats.mode(values_f)[0][0]
                else: 
                    yinzi_1d[it,f]=np.mean(values_f)
            print('a='+str(a)+',c='+str(c)+',max_id:'+str(max_id)+' '+str(it)+' is done'+' x:'+str(x_max-x_min+1)+' y:'+str(y_max-y_min+1))
            yinzi_1d[it,-2]=x_max-x_min+1
            yinzi_1d[it,-1]=y_max-y_min+1
        else:
            empty_index.append(it)
            filename='D:/yanshan_new/origion_shape/'+'a'+str(a)+'_c'+str(int(100*c)).zfill(3)+'/'+str(it)
            np.save(filename,np.zeros([1,1,nums_factors]))
            print(str(it)+' is done,but it is empty')
    
    df = pd.DataFrame(yinzi_1d)
    df.to_csv('D:/yanshan_new/origion_shape/'+'a'+str(a)+'_c'+str(int(100*c)).zfill(3)+'/yinzi1d.csv')
            

    #    plt.pyplot.imshow(img_it_resize[:,:,0])
    print('裁剪结束')

A=[100000]
C=[0.01,0.05,0.10,0.30,0.50]

base_path='D:/yanshan_new/tif_fid/'

for a in A:
    for c in C:
        tmp_path=base_path+'a'+str(a)+'_c'+str(int(100*c)).zfill(3)+'.tif'
        try:
            os.makedirs('D:/yanshan_new/origion_shape/'+'a'+str(a)+'_c'+str(int(100*c)).zfill(3))
        except FileExistsError as identifier:
            continue
        else:
            try:
                SUCardMake(tmp_path,a,c)
            except ValueError as identifier:
                continue
            
        
        

# band=len(totaldata[0][0])
# totaldata=np.reshape(totaldata,(row*col,band))


# 假如这是pro,显示最后的概率分布图
# pro=totaldata[:,4]
# pro_map=np.reshape(pro,(row,col))
# plt.pyplot.imshow(pro_map,cmap='bone')

# 如果要输入到arcgis当中，就把pro_map保存为tif
