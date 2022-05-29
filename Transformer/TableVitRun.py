

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
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score,auc,roc_curve,precision_recall_curve,f1_score, precision_score, recall_score
import torch.optim as optim
from VitNet import ViT
from DataLoad import DataLoad, BATCH_SIZE, Test_Batch_Size,TableDataLoad
from EarlyStop import EarlyStopping
# from bohb import BOHB
import configspace as cs
import  time
# from torchsummary import summary
from sklearn.preprocessing import label_binarize
import seaborn as sns
from sklearn import metrics
from matplotlib import pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def modeltrian(tp, EPOCH, LR, test_ratio, start, end, ncls, psize, depth, heads, mlp_dim, path):

    global NetPath


    data_train, data_test = TableDataLoad(tp, test_ratio, start, end, seed=80)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True)

    train_result_path = './/Result//Train//transfomertable.csv'
    test_result_path = './/Result//Test//transfomertable.csv'


    store_path = path

    model = ViT(
                    num_classes = ncls,
                    image_size = (end-start, 1),  # image size is a tuple of (height, width)
                    patch_size = (psize, 1),    # patch size is a tuple of (height, width)
                    dim = 2048, #1024 self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
                    depth = depth, #encoder和decoder的深度
                    heads = heads, #注意力机制的数量
                    mlp_dim = mlp_dim, #2048 encoder注意力机制后接多层全连接层
                    dropout = 0.1,
                    emb_dropout = 0.1
                    ).to(device)

    # summary(model, (1, 2000, 1), batch_size=1, device="cuda")

    criterion = nn.CrossEntropyLoss().to(device)  # 损失函数为焦损函数，多用于类别不平衡的多分类问题

    optimizer = optim.Adam(model.parameters(), lr=LR)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    early_stopping = EarlyStopping(patience=30, delta=1e-4, path=store_path, verbose=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=1, eps=1e-07,
                                                           patience=10)
    print("This is VitRun")
    print("Start Training!")  # 定义遍历数据集的次数
    with open(train_result_path, "w") as f1:
        with open(test_result_path, "w") as f2:
            f1.write("{},{},{}".format(("epoch"), ("loss"), ("acc")))  # 写入数据
            f1.write('\n')
            f2.write("{},{},{}".format(("epoch"), ("loss"), ("acc")))  # 写入数据
            f2.write('\n')
            for epoch in range(EPOCH):
                for i, data in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
                    sum_loss = []
                    model.train()  # 不训练
                    inputs, labels = data  # 输入和标签都等于data
                    inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
                    labels = Variable(labels).type(torch.LongTensor).to(device)  # batch y
                    output = model(inputs)  # cnn output
                    loss = criterion(output, labels)  # cross entropy loss
                    optimizer.zero_grad()  # clear gradients for this training step
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients
                    _, predicted = torch.max(output.data,1)  # _ , predicted这样的赋值语句，表示忽略第一个返回值，把它赋值给 _， 就是舍弃它的意思，预测值＝output的第一个维度
                    y_predicted = predicted.cpu().numpy()
                    label = labels.cpu().numpy()
                    acc = accuracy_score(label, y_predicted)
                    # print("trian:epoch = {:} Loss = {:.4f}  Acc= {:.4f}".format((epoch + 1), (loss.item()),(acc)))  # 训练次数，总损失，精确度
                    # f1.write("{:},{:.4f},{:.4f}".format((epoch + 1), (loss.item()), (acc)))  # 写入数据
                    # f1.write('\n')
                    # f1.flush()
                    sum_loss.append(loss.item())
                avg_loss = np.mean(sum_loss)


                with torch.no_grad():  # 无梯度
                    test_loss = []
                    for i, data in enumerate(test_loader):
                        model.eval()  # 不训练
                        inputs, labels = data  # 输入和标签都等于data
                        inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
                        labels = Variable(labels).type(torch.LongTensor).to(device)  # batch y
                        outputs = model(inputs)  # 输出等于进入网络后的输入
                        loss = criterion(outputs, labels)  # cross entropy loss
                        _, predicted = torch.max(outputs.data,1)  # _ , predicted这样的赋值语句，表示忽略第一个返回值，把它赋值给 _， 就是舍弃它的意思，预测值＝output的第一个维度 ，取得分最高的那个类 (outputs.data的索引号)
                        y_predicted = predicted.cpu().numpy()
                        label = labels.cpu().numpy()
                        acc = accuracy_score(label, y_predicted)
                        test_loss.append(loss.item())
                        # print("test:epoch = {:}   Acc= {:.4f}".format((epoch + 1) , (acc)))
                        # f2.write("{},{:.4f},{:.4f}".format((epoch + 1), (loss.item()), (acc)))  # 写入数据
                        # f2.write('\n')
                        # f2.flush()
            # 将每次测试结果实时写入acc.txt文件中



def modeltest(tp, test_ratio, start, end, ncls, psize, depth, heads, mlp_dim, path):
    # _, data_test = DataLoad('tou', test_ratio, start, end)
    data_train, data_test = TableDataLoad(tp, test_ratio, start, end, seed=80)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=Test_Batch_Size, shuffle=True)
    model = ViT(
                    num_classes = ncls,
                    image_size = (end-start, 1),  # image size is a tuple of (height, width)
                    patch_size = (psize, 1),    # patch size is a tuple of (height, width)
                    dim = 2048, #1024,
                    depth = depth,
                    heads = heads,
                    mlp_dim = mlp_dim, #2048, 1024
                    dropout = 0.1,
                    emb_dropout = 0.1
                    ).to(device)
    # store_path = './/model//all//transform'+'{}new.pt'.format(int(10-10*(test_ratio)))
    store_path = path
    model.load_state_dict(torch.load(store_path))
    acc_list = []
    for i, data in enumerate(test_loader):
        model.eval()  # 不训练
        inputs, labels = data  # 输入和标签都等于data
        inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
        labels = Variable(labels).type(torch.LongTensor).to(device)  # batch y
        outputs = model(inputs)  # 输出等于进入网络后的输入
        _, predicted = torch.max(outputs.data,1)  # _ , predicted这样的赋值语句，表示忽略第一个返回值，把它赋值给 _， 就是舍弃它的意思，预测值＝output的第一个维度 ，取得分最高的那个类 (outputs.data的索引号)
        y_predicted = predicted.cpu().numpy()
        label = labels.cpu().numpy()
        acc = accuracy_score(label, y_predicted)
        acc_list.append(acc)
    # print("Acc= {:.4f}".format(np.mean(acc_list)))
    return np.mean(acc_list)


class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes  # 类别数量，本例数据集类别为5
        self.labels = labels  # 类别标签

    def update(self, preds, labels):
        for p, t in zip(preds, labels):  # pred为预测结果，labels为真实标签
            self.matrix[p, t] += 1  # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1

    def summary(self):  # 计算指标函数
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_TP / n  # 总体准确率
        print("the model accuracy is ", acc)

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        # print("the model kappa is ", kappa)

        return str(acc)

    def plot(self):  # 绘制混淆矩阵
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc=' + self.summary() + ')')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


def model4AUCtest(tp, test_ratio, start, end, ncls, psize, depth, heads, mlp_dim, path):
    #_, data_test = DataLoad(tp, test_ratio, start, end)
    data_train, data_test = TableDataLoad(tp, test_ratio, start, end, seed=80)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=Test_Batch_Size, shuffle=True)
    model = ViT(
                    num_classes = ncls,
                    image_size = (end-start, 1),  # image size is a tuple of (height, width)
                    patch_size = (psize, 1),    # patch size is a tuple of (height, width)
                    dim = 2048, #1024,
                    depth = depth,
                    heads = heads,
                    mlp_dim = mlp_dim, #2048, 1024
                    dropout = 0.1,
                    emb_dropout = 0.1
                    ).to(device)
    # store_path = './/model//all//transform'+'{}new.pt'.format(int(10-10*(test_ratio)))
    store_path = path
    model.load_state_dict(torch.load(store_path))
    labels = [0, 1, 2, 3]
    # tomato_DICT = {'0': 'Bacterial_spot', '1': 'Early_blight', '2': 'healthy', '3': 'Late_blight', '4': 'Leaf_Mold'}
    # label = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=4, labels=labels)

    for i, data in enumerate(test_loader):
        model.eval()  # 不训练
        inputs, labels = data  # 输入和标签都等于data
        inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
        labels = Variable(labels).type(torch.LongTensor).to(device)  # batch y
        outputs = model(inputs)  # 输出等于进入网络后的输入
        y_proba = outputs.data.cpu().numpy()
        _, predicted = torch.max(outputs.data,1)  # _ , predicted这样的赋值语句，表示忽略第一个返回值，把它赋值给 _， 就是舍弃它的意思，预测值＝output的第一个维度 ，取得分最高的那个类 (outputs.data的索引号)
        y_predicted = predicted.cpu().numpy()
        label = labels.cpu().numpy()

        y_one_hot = label_binarize(label, classes=[0, 1, 2, 3])
        # ACC
        acc = accuracy_score(label, y_predicted)
        precis = precision_score(label, y_predicted, average='weighted')
        reca = recall_score(label, y_predicted, average='weighted')

        # labels_name = [0, 1, 2, 3]
        # arry = metrics.confusion_matrix(y_true=label, y_pred=y_predicted, labels=labels_name)  # 生成混淆矩阵

        confusion.update(y_predicted, label)


        # FPR,TPR
        false_positive_rate, true_positive_rate, _ = roc_curve(y_one_hot.ravel(), y_proba.ravel())
        # new_tpr
        mean_fpr = np.linspace(0, 1, 100)
        new_true_positive_rate = np.interp(mean_fpr, false_positive_rate, true_positive_rate)
        # AUC
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # Recall、Precision
        precision, recall, _ = precision_recall_curve(y_one_hot.ravel(), y_proba.ravel())
        # new_recall
        mean_recall = np.linspace(0, 1, 100)
        new_precision = np.interp(mean_recall, precision, recall)
        # new_precision = np.interp(mean_recall, recall, precision)
        # F1
        F1 = f1_score(label, y_predicted, average='weighted')
        # F2 = f1_score(y_test,y_pred,average='macro')
        # F3 = f1_score(y_test,y_pred,average='micro')

        # # runing_time
        # run_time = end - start
    # confusion.plot()
    # confusion.summary()

    return acc, precis, reca, F1, roc_auc #new_true_positive_rate, new_precision





if __name__ == "__main__":


    name = 'raw'

    sotre_path = './/model//Table//transformertable1125'+'{}.pt'.format(name)

    # modeltrian(tp=name, EPOCH=200, LR=0.0001, test_ratio=0.319,
    #                               start=0, end=400, ncls=4, psize=10, depth=3, heads=12, mlp_dim=1024,
    #                               path=sotre_path)  # depth=6, heads=10, 12, 14
    acc, precis, reca, F1, roc_auc = model4AUCtest(tp=name,test_ratio=0.319,
               start=0, end=400, ncls=4, psize=10, depth=3, heads=12, mlp_dim=1024, path=sotre_path)  # depth=6, heads=10, 12, 14

    print("acc:{}, precis:{}, recall:{}, F1:{}, auc:{}".format(acc, precis, reca, F1, roc_auc))

    #     acc = modeltest(tp=name, test_ratio=0.319, start=0, end=400,
    #                            ncls=4, psize=10, depth=3, heads=12, mlp_dim=1024, path=sotre_path)
    #     print(acc)
    #     # print(arry)
    #     # with open(result_path, "a") as file:
    #     #     file.write("{}, {}".format(
    #     #         name, acc))  # 写入数据
    #     #     file.write('\n')

    # list = [5,10,25,40,50,80,100]
    #
    # for name in list:
    #     sotre_path = './/model//Table//transformertable'+'{}.pt'.format(name)
    #
    #
    #     result_path = './/Result//Table//transformertabale'+'.csv'
    #     modeltrian(tp='raw', EPOCH=200, LR=0.0001, test_ratio=0.319,
    #                                   start=0, end=400, ncls=4, psize=name, depth=3, heads=12, mlp_dim=12,
    #                                   path=sotre_path)  # depth=6, heads=10, 12, 14
    #     acc = modeltest(tp='raw', test_ratio=0.319, start=0, end=400,
    #                            ncls=4, psize=name, depth=3, heads=12, mlp_dim=12, path=sotre_path)
    #     print(acc)
    #
    #     with open(result_path, "a") as file:
    #         file.write("{}, {}, {}".format(
    #             'psize', acc, name))  # 写入数据
    #         file.write('\n')

