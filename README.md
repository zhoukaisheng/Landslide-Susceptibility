# Landslide-Susceptibility


SlopeCardMake.py 输入tif格式的影响因子图层与以fid为灰度值的斜坡单元图层,输出fid.npy的斜坡单元卡片与1D因子。
SlopeCardResize.py 将斜坡单元卡片rezise为统一大小并得到一个npy文件。
data_read.py 数据读取与增强。
nni_trial 借助nni实现模型的超参数优化，并进行滑坡易发性评价。

First,installing Requirements
Second,cd nni_trial
Last,nnictl create --config config.yml

Requirements:keras/nni/pandas/numpy/matplotlib
