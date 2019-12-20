import numpy as np 
import matplotlib.pyplot as plt 
import imageio

def get_SU_label(label_row_col,su_fid,a,c):
    #1 tt_nolandslide 0
    #2 tr_nolandslide 1
    #3 tt_landslide 2
    #4 tr_landslide 3
    max_id = np.max(su_fid[su_fid != 65535])
    id_count=np.zeros([max_id,4])
    for i in range(4):
        locs=label_row_col[i]
        # print(locs)
        # print(su_fid.shape)
        fids=su_fid[locs]
        # print(locs)
        # print(fids)
        # plt.figure
        # plt.imshow(su_fid)
        # plt.scatter(locs[1],locs[0])
        # plt.show()
        try:
            id_count[fids[fids!=-32768],i]=id_count[fids[fids!=-32768],i]+1
        except IndexError as identifier:
            id_count[fids[fids!=65535],i]=id_count[fids[fids!=65535],i]+1
        else:
            pass
        
        # nums=loc.shape[0]
        # for i in range(nums):
        #     loc=locs[i,:]
        #     fid=su_fid[loc]
        # print(id_count)
        # print(np.count_nonzero(id_count[:,i]))
    label=np.zeros([max_id,1])
    label[id_count[:,2]!=0]=1
    label[id_count[:,3]!=0]=1
    tr=np.concatenate([np.where(id_count[:,1]!=0),np.where(id_count[:,3]!=0)],axis=1)
    tt=np.concatenate([np.where(id_count[:,0]!=0),np.where(id_count[:,2]!=0)],axis=1)
    tr_name='D:/yanshan_new/tr_tt_npy_and_label/tr_'+'a'+str(a)+'_c'+str(int(100*c)).zfill(3)+'.npy'
    tt_name='D:/yanshan_new/tr_tt_npy_and_label/tt_'+'a'+str(a)+'_c'+str(int(100*c)).zfill(3)+'.npy'
    label_name='D:/yanshan_new/tr_tt_npy_and_label/label_'+'a'+str(a)+'_c'+str(int(100*c)).zfill(3)+'.npy'
    np.save(tr_name,tr)
    np.save(tt_name,tt)
    np.save(label_name,label)
    print(tr.shape,tt.shape)
    return tr,tt,label


def get_row_col(label):
    row_col=[]
    for i in range(1,5):
        row_col.append(np.where(label==i))
    return row_col



A=[10000,50000,100000,200000,300000]
C=[0.01,0.05,0.10,0.30,0.50]



base_path='D:/yanshan_new/tif_fid/'
label=imageio.imread('D:/yanshan_new/tr_and_tt.tif')
label_loc=get_row_col(label)
# print(label_loc)
for a in A:
    for c in C:
        tmp_path=base_path+'a'+str(a)+'_c'+str(int(100*c)).zfill(3)+'.tif'
        fid=imageio.imread(tmp_path)
        id_count=get_SU_label(label_loc,fid,a,c)
        # print(id_count)

