from Chemometrics import SVM, PLS_DA1
from CNN import CNN
from AE import SAE1
from DataLoad import DataLoad, BaseDataLoad, TableDataLoad
import numpy as np

if __name__ == '__main__':

    # tp_list7 = ['tou','7msc', '7sg', '7snv']
    # # tp_list7 = ['7msc']
    # # tp_list18 = ['18msc', '18sg', '18snv', 'all']
    # ratio_list = [0.3, 0.7]
    # # test_ratio = 0.3
    #
    # for test_ratio in ratio_list:
    #     for tp in tp_list7:
    #         result_path = './/Result//Drug7//baselineprosesing' + '{}.csv'.format(10 * test_ratio)
    #         X_train, X_test, y_train, y_test = BaseDataLoad(tp=tp, test_ratio=test_ratio, start=0, end=2074)
    #         SVMACC, SVMTrain_time, SVMTest_times = SVM(train_x=X_train, train_y=y_train, test_x=X_test, test_y=y_test)
    #         print('svm ok')
    #         PLSACC, PLSTrain_time, PLSTest_times = PLS_DA1(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    #         print('pls ok and acc:{}'.format(PLSACC))
    #         SAEACC, SAETrain_time, SAETest_times = SAE1(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    #         print('sae ok')
    #         CNNACC, CNNTrain_time, CNNTest_times = CNN(tp, test_ratio, 16, 200, nls=18)
    #         print('cnn ok')
    #
    #         with open(result_path, "a") as file:
    #                 file.write("svm:tp:{}, test_ratio:{}, ACC:{:6f}".format(tp, test_ratio, SVMACC))  # 写入数据
    #                 file.write('\n')#数据
    #                 file.write("pls:tp:{}, test_ratio:{}, ACC:{:6f}".format(tp, test_ratio, PLSACC))  # 写入数据
    #                 file.write('\n')  # 数据
    #                 file.write("sae:tp:{}, test_ratio:{}, ACC:{:6f}".format(tp, test_ratio, SAEACC))  # 写入数据
    #                 file.write('\n')  # 数据
    #                 file.write("cnn:tp:{}, test_ratio:{}, ACC:{:6f}".format(tp, test_ratio, CNNACC))  # 写入数据
    #                 file.write('\n')  # 数据

    tp_list = ['raw']#,'MSC', 'SG', 'SNV']
    # tp_list7 = ['7msc']
    # tp_list18 = ['18msc', '18sg', '18snv', 'all']

    test_ratio = 0.319

    for tp in tp_list:
        result_path = './/Result//Table//Tablebaselineprosesing' + '{}.csv'.format(10 * test_ratio)
        X_train, X_test, y_train, y_test = TableDataLoad(tp=tp, test_ratio=test_ratio, start=0, end=404, seed=80)
        SVMACC, SVMTrain_time, SVMTest_times = SVM(train_x=X_train, train_y=y_train, test_x=X_test, test_y=y_test)
        print('svm ok')
        print('svm ok and acc:{}'.format(SVMACC))
        PLSACC, PLSTrain_time, PLSTest_times = PLS_DA1(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        print('pls ok and acc:{}'.format(PLSACC))
        SAEACC, SAETrain_time, SAETest_times = SAE1(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        print('sae ok:{}'.format(SAEACC))
        # CNNACC, CNNTrain_time, CNNTest_times = CNN(tp, test_ratio, 16, 200, nls=4)
        # print('cnn ok')
        # print(CNNTrain_time+CNNTest_times)

        # with open(result_path, "a") as file:
        #         # file.write("svm:tp:{}, test_ratio:{}, ACC:{:6f}".format(tp, test_ratio, SVMACC))  # 写入数据
        #         # file.write('\n')#数据
        #         # file.write("pls:tp:{}, test_ratio:{}, ACC:{:6f}".format(tp, test_ratio, PLSACC))  # 写入数据
        #         # file.write('\n')  # 数据
        #         # file.write("sae:tp:{}, test_ratio:{}, ACC:{:6f}".format(tp, test_ratio, SAEACC))  # 写入数据
        #         # file.write('\n')  # 数据
        #         file.write("cnnstan:tp:{}, test_ratio:{}, ACC:{:6f}".format(tp, test_ratio, CNNACC))  # 写入数据
        #         file.write('\n')  # 数据


