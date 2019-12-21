import numpy as np
import matplotlib.pyplot as plt 
import imageio

def get_row_col(label):
    row_col=[]
    for i in range(1,5):
        row_col.append(np.where(label==i))
    return row_col

def read_factor_data():
    yinzi_name = ['altitude', 'aspect', 'fault', 'ndvi', 'plan', 'profile', 'rainfall',
        'river', 'road', 'slope', 'spi', 'sti', 'twi', 'soil', 'landuse', 'lithology']

    row = 2636
    col = 2206
    nums_factors = len(yinzi_name)
    img = np.zeros((row, col, nums_factors))
    i = 0
    yinzi_type = []
    for i in range(nums_factors):
        # print(i)
        name = yinzi_name[i]
        PATH = 'C:/Users/75129/Desktop/mypy/demo_all/tif_yanshan/'

        ttt = imageio.imread(PATH+name+'.tif')
    #    if yinzi_type_ori[i]=='str':
    #        plt.pyplot.figure()
    #        plt.pyplot.imshow(img[:,:,i])
        ttt[ttt == -32768] = 0
    #    ttt[ttt==-3.402823e+038]=0
        img[:, :, i] = ttt
    #    img_i=load_data[name]
    #    ma=np.max(img_i)
    #    mi=np.min(img_i)
    #    img[:,:,i]=(img_i-mi)/(ma-mi)
        i = i+1
    # pca = PCA(n_c
    return img
def get_3d_x_save_memory(factors,window_size):
    # windows_size must be Odd number
    factors_3d=np.zeros([factors.shape[0],factors.shape[1],factors.shape[2]])
    index=np.arange(0,window_size*window_size)
    index=index.reshape([window_size,window_size])
    # print(index)
    loc_x=[]
    loc_y=[]
    r=(window_size-1)/2
    for i in range(window_size*window_size):
        loc=np.where(index==i)
        x=loc[0]
        y=loc[1]
        x=r-x
        y=r-y
        loc_x.append(x)
        loc_y.append(y)
    # print(loc_x)
    # print(loc_y)
    for i in range(window_size*window_size):
        print(i)
        x_start=int(max(loc_x[i],0))
        x_end=int(min(loc_x[i],-0))
        y_start=int(max(loc_y[i],0))
        y_end=int(min(loc_y[i],-0))

        x_start_3d=-x_end
        x_end_3d=-x_start
        y_start_3d=-y_end
        y_end_3d=-y_start
        # print(x_start,x_end)
        # print(y_start,y_end)
        if x_end==0:
            x_end=None
        if y_end==0:
            y_end=None
        if x_end_3d==0:
            x_end_3d=None 
        if y_end_3d==0:
            y_end_3d=None 

        tmp=factors[x_start_3d:x_end_3d,y_start_3d:y_end_3d,:]
        # print(tmp.shape)
        # print(factors_3d[i,x_start_3d:x_end_3d,y_start_3d:y_end_3d,:].shape)
        factors_3d[x_start:x_end,y_start:y_end,:]=tmp
        np.save('D:/yanshan_new/pixel_traing_data/factors3d_'+str(window_size)+'_'+str(i)+'.npy',factors_3d)
    # factors_3d=factors_3d.reshape([factors.shape[0],factors.shape[1],window_size,window_size,factors.shape[2]])
        print(factors_3d.shape)
    return None

def get_3d_x(factors,window_size):
    # windows_size must be Odd number
    factors_3d=np.zeros([window_size*window_size,factors.shape[0],factors.shape[1],factors.shape[2]])
    index=np.arange(0,window_size*window_size)
    index=index.reshape([window_size,window_size])
    # print(index)
    loc_x=[]
    loc_y=[]
    r=(window_size-1)/2
    for i in range(window_size*window_size):
        loc=np.where(index==i)
        x=loc[0]
        y=loc[1]
        x=r-x
        y=r-y
        loc_x.append(x)
        loc_y.append(y)
    # print(loc_x)
    # print(loc_y)
    for i in range(window_size*window_size):
        x_start=int(max(loc_x[i],0))
        x_end=int(min(loc_x[i],-0))
        y_start=int(max(loc_y[i],0))
        y_end=int(min(loc_y[i],-0))

        x_start_3d=-x_end
        x_end_3d=-x_start
        y_start_3d=-y_end
        y_end_3d=-y_start
        # print(x_start,x_end)
        # print(y_start,y_end)
        if x_end==0:
            x_end=None
        if y_end==0:
            y_end=None
        if x_end_3d==0:
            x_end_3d=None 
        if y_end_3d==0:
            y_end_3d=None 

        tmp=factors[x_start_3d:x_end_3d,y_start_3d:y_end_3d,:]
        # print(tmp.shape)
        # print(factors_3d[i,x_start_3d:x_end_3d,y_start_3d:y_end_3d,:].shape)
        factors_3d[i,x_start:x_end,y_start:y_end,:]=tmp
    factors_3d=factors_3d.reshape([window_size,window_size,factors.shape[0],factors.shape[1],factors.shape[2]])
    factors_3d=np.swapaxes(factors_3d,0,2)
    factors_3d=np.swapaxes(factors_3d,1,3)
    print(factors_3d.shape)
    return factors_3d

def get_tr_tt_data_3d(data_3d,tr_tt_img):
    #1 tt_nolandslide 0
    #2 tr_nolandslide 1
    #3 tt_landslide 2
    #4 tr_landslide 3
    #data_3d.shape = (row,col,window_size,window_size,channel)
    data_3d=data_3d.reshape([data_3d.shape[0]*data_3d.shape[1],data_3d.shape[2],data_3d.shape[3],data_3d.shape[4]])
    tr_tt_img=tr_tt_img.reshape([tr_tt_img.shape[0]*tr_tt_img.shape[1],])
    locs=get_row_col(tr_tt_img)
    for i in range(4):
        loc=locs[i]
        # print(loc)
        tmp_data=data_3d[loc,:,:,:]
        tmp_data=np.squeeze(tmp_data)
        print(tmp_data.shape)
        if i==0:
            tt_x=tmp_data
            tt_y=np.zeros([tmp_data.shape[0],])
        elif i==1:
            tr_x=tmp_data
            tr_y=np.zeros([tmp_data.shape[0],])
        elif i==2:
            tt_x=np.concatenate((tt_x,tmp_data),axis=0)
            tt_y=np.concatenate((tt_y,np.ones([tmp_data.shape[0],])),axis=0)
        elif i==3:
            tr_x=np.concatenate((tr_x,tmp_data),axis=0)
            tr_y=np.concatenate((tr_y,np.ones([tmp_data.shape[0],])),axis=0)
    # print(tt_x.shape)
    # print(tt_y)
    return tr_x,tr_y,tt_x,tt_y

def get_tr_tt_data_3d_save_memory(data_3d_path,tr_tt_img,window_size):
    #1 tt_nolandslide 0
    #2 tr_nolandslide 1
    #3 tt_landslide 2
    #4 tr_landslide 3
    #data_3d.shape = (row,col,channel)
    train_x=[]
    # train_y=[]
    test_x=[]
    # test_y=[]
    tr_tt_img=tr_tt_img.reshape([tr_tt_img.shape[0]*tr_tt_img.shape[1],])
    locs=get_row_col(tr_tt_img)
    for i in range(window_size*window_size):
        data_3d=np.load(data_3d_path+'factors3d_'+str(window_size)+'_'+str(i)+'.npy')
        # data_3d=np.load('D:/yanshan_new/pixel_traing_data/factors3d_'+str(window_size)+'_'+str(i)+'.npy')
        data_3d=data_3d.reshape([data_3d.shape[0]*data_3d.shape[1],data_3d.shape[2]])
        
        
        for i in range(4):
            loc=locs[i]
            # print(loc)
            tmp_data=data_3d[loc,:]
            tmp_data=np.squeeze(tmp_data)
            # print(tmp_data.shape)
            if i==0:
                tt_x=tmp_data
                tt_y=np.zeros([tmp_data.shape[0],])
            elif i==1:
                tr_x=tmp_data
                tr_y=np.zeros([tmp_data.shape[0],])
            elif i==2:
                tt_x=np.concatenate((tt_x,tmp_data),axis=0)
                tt_y=np.concatenate((tt_y,np.ones([tmp_data.shape[0],])),axis=0)
            elif i==3:
                tr_x=np.concatenate((tr_x,tmp_data),axis=0)
                tr_y=np.concatenate((tr_y,np.ones([tmp_data.shape[0],])),axis=0)
        train_x.append(tr_x)
        test_x.append(tt_x)
    train_x=np.array(train_x)
    test_x=np.array(test_x)

    train_x=np.swapaxes(train_x,0,1)
    test_x=np.swapaxes(test_x,0,1)

    train_x=train_x.reshape([train_x.shape[0],window_size,window_size,train_x.shape[2]])
    test_x=test_x.reshape([test_x.shape[0],window_size,window_size,test_x.shape[2]])
    print(train_x.shape)
    print(test_x.shape)
    plt.imshow(train_x[10,:,:,0])
    plt.show()
    # print(tt_x.shape)
    # print(tt_y)
    return train_x,tr_y,test_x,tt_y




if __name__ == '__main__':
    factors=read_factor_data()
    window_size=7
    print('begin get_3d_x')
    factors_3d=get_3d_x(factors,window_size)
    print('end get_3d_x')
    label=imageio.imread('D:/yanshan_new/tr_and_tt.tif')
    data_3d_path='D:/yanshan_new/pixel_traing_data/'
    tr_x,tr_y,tt_x,tt_y=get_tr_tt_data_3d(factors_3d,label)
    # tr_x,tr_y,tt_x,tt_y=get_tr_tt_data_3d(factors_3d,label)
    # save memory
    np.save('D:/yanshan_new/pixel_traing_data/factors_3d_'+str(window_size)+'.npy',factors_3d)
    np.save('D:/yanshan_new/pixel_traing_data/tr_x_'+str(window_size)+'.npy',tr_x)
    np.save('D:/yanshan_new/pixel_traing_data/tr_y_'+str(window_size)+'.npy',tr_y)
    np.save('D:/yanshan_new/pixel_traing_data/tt_x_'+str(window_size)+'.npy',tt_x)
    np.save('D:/yanshan_new/pixel_traing_data/tt_y_'+str(window_size)+'.npy',tt_y)


