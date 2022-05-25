from Chemometrics import SVM, PLS_DA1
from CNN import CNN
from AE import SAE1
from DataLoad import DataLoad, BaseDataLoad
import numpy as np

if __name__ == '__main__':

    ratio_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    for test_ratio in ratio_list:

        acc_diff_ratio_svm = []
        Time1_diff_ratio_svm = []
        Time2_diff_ratio_svm = []

        acc_diff_ratio_pls = []
        Time1_diff_ratio_pls = []
        Time2_diff_ratio_pls = []

        acc_diff_ratio_CNN = []
        Time1_diff_ratio_CNN = []
        Time2_diff_ratio_CNN = []

        acc_diff_ratio_SAE = []
        Time1_diff_ratio_SAE = []
        Time2_diff_ratio_SAE = []

        for idex in range(5):
            result_path = './/Result//Drug7//baseline' + '{}.csv'.format(10 * test_ratio)
            X_train, X_test, y_train, y_test = BaseDataLoad(tp='tou', test_ratio=test_ratio, start=0, end=2074)
            SVMACC, SVMTrain_time, SVMTest_times = SVM(train_x=X_train, train_y=y_train, test_x=X_test, test_y=y_test)
            print('svm ok')
            PLSACC, PLSTrain_time, PLSTest_times = PLS_DA1(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            print('pls ok')
            # SAEACC, SAETrain_time, SAETest_times = SAE1(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            # print('sae ok')
            # CNNACC, CNNTrain_time, CNNTest_times = CNN(test_ratio, 16, 200)
            # print('cnn ok')

            acc_diff_ratio_svm.append(SVMACC)
            acc_diff_ratio_pls.append(PLSACC)
            # acc_diff_ratio_SAE.append(SAEACC)
            # acc_diff_ratio_CNN.append(CNNACC)

            Time1_diff_ratio_svm.append(SVMTrain_time)
            Time1_diff_ratio_pls.append(PLSTrain_time)
            # Time1_diff_ratio_SAE.append(SAETrain_time)
            # Time1_diff_ratio_CNN.append(CNNTrain_time)

            Time2_diff_ratio_svm.append(SVMTest_times)
            Time2_diff_ratio_pls.append(PLSTest_times)
            # Time2_diff_ratio_SAE.append(SAETest_times)
            # Time2_diff_ratio_CNN.append(CNNTest_times)

        with open(result_path, "a") as file:
                file.write("svm:ACC:{}, STD:{}, TIME1:{:.6f}, TIME2:{:6f}".format(
                     np.mean(acc_diff_ratio_svm), np.std(acc_diff_ratio_svm), np.mean(Time1_diff_ratio_svm),  np.mean(Time1_diff_ratio_svm)))  # 写入数据
                file.write('\n')
                file.write("pls:ACC:{}, STD:{}, TIME1:{:.6f}, TIME2:{:6f}".format(
                    np.mean(acc_diff_ratio_pls), np.std(acc_diff_ratio_pls), np.mean(Time1_diff_ratio_pls),
                    np.mean(Time1_diff_ratio_pls)))  # 写入数据
                file.write('\n')
                print("pls:ACC:{}, STD:{}, TIME1:{:.6f}, TIME2:{:6f}".format(
                    np.mean(acc_diff_ratio_pls), np.std(acc_diff_ratio_pls), np.mean(Time1_diff_ratio_pls),
                    np.mean(Time1_diff_ratio_pls)))
                # file.write("sae:ACC:{}, STD:{}, TIME1:{:.6f}, TIME2:{:6f}".format(
                #     np.mean(acc_diff_ratio_SAE), np.std(acc_diff_ratio_SAE), np.mean(Time1_diff_ratio_SAE),
                #     np.mean(Time1_diff_ratio_SAE)))  # 写入数据
                # file.write('\n')
                # file.write("cnn:ACC:{}, STD:{}, TIME1:{:.6f}, TIME2:{:6f}".format(
                #     np.mean(acc_diff_ratio_CNN), np.std(acc_diff_ratio_CNN), np.mean(Time1_diff_ratio_CNN),
                #     np.mean(Time1_diff_ratio_CNN)))  # 写入数据
                # file.write('\n')
