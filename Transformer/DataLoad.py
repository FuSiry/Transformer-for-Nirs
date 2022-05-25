"""
    -*- coding: utf-8 -*-
    @Time   :2021/011/12 13:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/Transformer-for-Nirs
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""

import numpy as np
# import  pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler

# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


BATCH_SIZE = 32
Test_Batch_Size = 99 #32
random_state = 80

#139, 208,256,  415, 484,553
#自定义加载数据集
class MyDataset(Dataset):
    def __init__(self,specs,labels):
        self.specs = specs
        self.labels = labels

    def __getitem__(self, index):
        spec,target = self.specs[index],self.labels[index]
        return spec,target

    def __len__(self):
        return len(self.specs)

# ###定义是否需要标准化
def ZspPocess(X_train, X_test,y_train,y_test,need=True): #True:需要标准化，Flase：不需要标准化
    if (need == True):
        # X_train_Nom = scale(X_train)
        # X_test_Nom = scale(X_test)
        standscale = StandardScaler()
        X_train_Nom = standscale.fit_transform(X_train)
        X_test_Nom = standscale.transform(X_test)

        X_train_Nom = X_train_Nom[:, np.newaxis, :]
        X_test_Nom = X_test_Nom[:, np.newaxis, :]
        data_train = MyDataset(X_train_Nom, y_train)
        ##使用loader加载测试数据
        data_test = MyDataset(X_test_Nom, y_test)
        return data_train, data_test
    else:
        X_train = X_train[:, np.newaxis, :]  # （483， 1， 2074）
        X_test = X_test[:, np.newaxis, :]
        data_train = MyDataset(X_train, y_train)
        ##使用loader加载测试数据
        data_test = MyDataset(X_test, y_test)
        return data_train, data_test
#
###定义是否需要标准化
def ZspPocessnew(X_train, X_test,y_train,y_test,need=True): #True:需要标准化，Flase：不需要标准化
    if (need == True):
        # X_train_Nom = scale(X_train)
        # X_test_Nom = scale(X_test)
        standscale = StandardScaler()
        X_train_Nom = standscale.fit_transform(X_train)
        X_test_Nom = standscale.transform(X_test)

        X_train_Nom = X_train_Nom[:, np.newaxis, :]
        X_test_Nom = X_test_Nom[:, np.newaxis, :]
        X_train_Nom = X_train_Nom[:, :, :, np.newaxis]
        X_test_Nom = X_test_Nom[:, :, :, np.newaxis]
        data_train = MyDataset(X_train_Nom, y_train)
        ##使用loader加载测试数据
        data_test = MyDataset(X_test_Nom, y_test)
        return data_train, data_test
    else:

        X_train = X_train[:, np.newaxis, :]  # （483， 1， 2074）
        X_test = X_test[:, np.newaxis, :]
        X_train = X_train[:, :, :, np.newaxis]
        X_test = X_test[:, :, :, np.newaxis]
        data_train = MyDataset(X_train, y_train)
        ##使用loader加载测试数据
        data_test = MyDataset(X_test, y_test)
        return data_train, data_test


##############################无表头数据##################################
def plotspc(x_col, data_x, tp):
    # figsize = 5, 3
    figsize = 8, 5.5
    figure, ax = plt.subplots(figsize=figsize,dpi=500)
    # ax = plt.figure(figsize=(5,3))
    x_col = x_col[::-1]  # 数组逆序
    y_col = np.transpose(data_x)
    plt.plot(x_col, y_col )
    plt.tick_params(labelsize=12)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    font = {'weight': 'normal',
            'size': 16,
            }
    plt.xlabel("Wavenumber/$\mathregular{cm^{-1}}$", font)
    plt.ylabel("Absorbance", font)
    # plt.title("The spectrum of the {} dataset".format(tp), fontweight="semibold", fontsize='x-large')
    plt.show()
    plt.tick_params(labelsize=23)


def DataLoad(tp, test_ratio, start, end):

    data_y7_path = './/Data//头孢.csv'
    data_y7 = np.loadtxt(open(data_y7_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    datax1_path = './/Data//头孢.csv'
    datax2_path = './/Data//苯妥英钠.csv'
    datax1 = np.loadtxt(open(datax1_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    datax2 = np.loadtxt(open(datax2_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    data_y18 = np.concatenate((datax1, datax2), axis=0)

    if tp=="tou":
        X_path = './Data/4_class_not/axe.csv'
        data_path = './/Data//头孢.csv'
        data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        col = np.loadtxt(open(X_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        x_col = col[start:end]
        data_x = data[0:, start:end]
        data_y = data[0:, -1]
    elif tp=="ben":
        X_path = './Data/4_class_not/axe.csv'
        data_path = './/Data//苯妥英钠.csv'
        col = np.loadtxt(open(X_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        x_col  = col[start:end]
        data_x = data[0:, start:end]
        data_y = data[0:, -1]
    elif tp == 'all':
        X_path = './Data/4_class_not/axe.csv'
        data1_path = './/Data//头孢.csv'
        data2_path = './/Data//苯妥英钠.csv'
        data1 = np.loadtxt(open(data1_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        data2 = np.loadtxt(open(data2_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        col = np.loadtxt(open(X_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        data = np.concatenate((data1, data2), axis=0)
        x_col = col[start:end]
        data_x = data[0:, start:end]
        data_y = data[0:, -1]
    elif tp == '7msc':
        X_path = './Data/4_class_not/axe.csv'
        data_path = './/Data//头孢MSC.csv'
        data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        col = np.loadtxt(open(X_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        x_col = col[start:end]
        data_x = data[0:, start:end]
        data_y = data_y7[0:, -1]
    elif tp == '7sg':
        X_path = './Data/4_class_not/axe.csv'
        data_path = './/Data//头孢SG.csv'
        data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        col = np.loadtxt(open(X_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        x_col = col[start:end]
        data_x = data[0:, start:end]
        data_y = data_y7[0:, -1]
    elif tp == '7snv':
        X_path = './Data/4_class_not/axe.csv'
        data_path = './/Data//头孢SNV.csv'
        data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        col = np.loadtxt(open(X_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        x_col = col[start:end]
        data_x = data[0:, start:end]
        data_y = data_y7[0:, -1]
    elif tp == '18msc':
        X_path = './Data/4_class_not/axe.csv'
        data_path = './/Data//ALLMSC.csv'
        data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        col = np.loadtxt(open(X_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        x_col = col[start:end]
        data_x = data[0:, start:end]
        data_y = data_y18[0:, -1]
    elif tp == '18sg':
        X_path = './Data/4_class_not/axe.csv'
        data_path = './/Data//ALLSG.csv'
        data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        col = np.loadtxt(open(X_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        x_col = col[start:end]
        data_x = data[0:, start:end]
        data_y = data_y18[0:, -1]
    elif tp == '18snv':
        X_path = './Data/4_class_not/axe.csv'
        data_path = './/Data//ALLSNV.csv'
        data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        col = np.loadtxt(open(X_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        x_col = col[start:end]
        data_x = data[0:, start:end]
        data_y = data_y18[0:, -1]
    else:
        print("no dataset")

    plotspc(x_col, data_x[:, start:end], tp)

    x_data = np.array(data_x)
    y_data = np.array(data_y)
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_ratio,random_state=random_state)

    print('训练集规模：{}'.format(len(X_train[:,0])))
    print('测试集规模：{}'.format(len(X_test[:,0])))

    #data_train, data_test = ZspPocessnew(X_train, X_test,y_train,y_test,need=True)  #for transformer :false only  used in proseesing comparsion
    #data_train, data_test = ZspPocess(X_train, X_test,y_train,y_test,need=False) #for cnn :false only used in proseesing comparsion

    return data_train, data_test

def TableDataLoad(tp, test_ratio, start, end, seed):

    # global data_x
    data_path = './/Data//table.csv'
    Rawdata = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    table_random_state = seed

    if tp =='raw':
        data_x = Rawdata[0:, start:end]

        # x_col = np.linspace(0, 400, 400)
    if tp =='SG':
        SGdata_path = './/Data//TableSG.csv'
        data = np.loadtxt(open(SGdata_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        data_x = data[0:, start:end]
    if tp =='SNV':
        SNVata_path = './/Data//TableSNV.csv'
        data = np.loadtxt(open(SNVata_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        data_x = data[0:, start:end]
    if tp == 'MSC':
        MSCdata_path = './/Data//TableMSC.csv'
        data = np.loadtxt(open(MSCdata_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        data_x = data[0:, start:end]
    data_y = Rawdata[0:, -1]
    x_col = np.linspace(7400, 10507, 400)
    plotspc(x_col, data_x[:, :], tp=0)

    x_data = np.array(data_x)
    y_data = np.array(data_y)
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_ratio,random_state=table_random_state)
    # return X_train, X_test, y_train, y_test
    # standscale = StandardScaler()
    # X_train_Nom = standscale.fit_transform(X_train)
    # X_test_Nom = standscale.transform(X_test)  ##no standard for proseesing comprasion
    # return  X_train_Nom, X_test_Nom, y_train, y_test

    print('训练集规模：{}'.format(len(X_train[:,0])))
    print('测试集规模：{}'.format(len(X_test[:,0])))

    data_train, data_test = ZspPocessnew(X_train, X_test,y_train,y_test,need=False)  #for transformer :false only  used in proseesing comparsion
    #data_train, data_test = ZspPocess(X_train, X_test,y_train,y_test,need=True) #for cnn :false only used in proseesing comparsion
    return data_train, data_test

def BaseDataLoad(tp, test_ratio, start, end):
    data_y7_path = './/Data//头孢.csv'
    data_y7 = np.loadtxt(open(data_y7_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    datax1_path = './/Data//头孢.csv'
    datax2_path = './/Data//苯妥英钠.csv'
    datax1 = np.loadtxt(open(datax1_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    datax2 = np.loadtxt(open(datax2_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    data_y18 = np.concatenate((datax1, datax2), axis=0)

    if tp=="tou":
        X_path = './Data/4_class_not/axe.csv'
        data_path = './/Data//头孢.csv'
        data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        col = np.loadtxt(open(X_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        x_col = col[start:end]
        data_x = data[0:, start:end]
        data_y = data[0:, -1]
    elif tp=="ben":
        X_path = './Data/4_class_not/axe.csv'
        data_path = './/Data//苯妥英钠.csv'
        col = np.loadtxt(open(X_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        x_col  = col[start:end]
        data_x = data[0:, start:end]
        data_y = data[0:, -1]
    elif tp == 'all':
        X_path = './Data/4_class_not/axe.csv'
        data1_path = './/Data//头孢.csv'
        data2_path = './/Data//苯妥英钠.csv'
        data1 = np.loadtxt(open(data1_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        data2 = np.loadtxt(open(data2_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        col = np.loadtxt(open(X_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        data = np.concatenate((data1, data2), axis=0)
        x_col = col[start:end]
        data_x = data[0:, start:end]
        data_y = data[0:, -1]
    elif tp == '7msc':
        X_path = './Data/4_class_not/axe.csv'
        data_path = './/Data//头孢MSC.csv'
        data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        col = np.loadtxt(open(X_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        x_col = col[start:end]
        data_x = data[0:, start:end]
        data_y = data_y7[0:, -1]
    elif tp == '7sg':
        X_path = './Data/4_class_not/axe.csv'
        data_path = './/Data//头孢SG.csv'
        data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        col = np.loadtxt(open(X_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        x_col = col[start:end]
        data_x = data[0:, start:end]
        data_y = data_y7[0:, -1]
    elif tp == '7snv':
        X_path = './Data/4_class_not/axe.csv'
        data_path = './/Data//头孢SNV.csv'
        data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        col = np.loadtxt(open(X_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        x_col = col[start:end]
        data_x = data[0:, start:end]
        data_y = data_y7[0:, -1]
    elif tp == '18msc':
        X_path = './Data/4_class_not/axe.csv'
        data_path = './/Data//ALLMSC.csv'
        data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        col = np.loadtxt(open(X_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        x_col = col[start:end]
        data_x = data[0:, start:end]
        data_y = data_y18[0:, -1]
    elif tp == '18sg':
        X_path = './Data/4_class_not/axe.csv'
        data_path = './/Data//ALLSG.csv'
        data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        col = np.loadtxt(open(X_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        x_col = col[start:end]
        data_x = data[0:, start:end]
        data_y = data_y18[0:, -1]
    elif tp == '18snv':
        X_path = './Data/4_class_not/axe.csv'
        data_path = './/Data//ALLSNV.csv'
        data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        col = np.loadtxt(open(X_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        x_col = col[start:end]
        data_x = data[0:, start:end]
        data_y = data_y18[0:, -1]

    else:
        print("no dataset")
    start = 0
    # for i in range(40):
    #     plotspc(x_col[start:start+40], data_x[3:4, start:start+40], tp)
    #     start = start+40

    x_data = np.array(data_x)
    y_data = np.array(data_y)
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_ratio,random_state=random_state)

    standscale = StandardScaler()
    X_train_Nom = standscale.fit_transform(X_train)
    X_test_Nom = standscale.transform(X_test)  ##no standard for proseesing comprasion

    print('训练集规模：{}'.format(len(X_train[:,0])))
    print('测试集规模：{}'.format(len(X_test[:,0])))

    return X_train_Nom, X_test_Nom, y_train, y_test
    # return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # ratio_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # for test_ratio in ratio_list:
    #     data_train, data_test = DataLoad("tou", test_ratio, 0, 2074)

    data_train, data_test = DataLoad("ben", 0.2, 0, 2074)
    #TableDataLoad('raw', 0.2, 0, 400, 80)
    # data_train, data_test = BaseDataLoad("all", 0.3, 0, 2000)
    # a = torch.randn(1, 18, 4)
    # b = torch.zeros(1,18,2)
    # print(a)
    #
    # print(b)
    # TableDataLoad(tp='SG', test_ratio=0.319, start=0, end=404, seed=80)
