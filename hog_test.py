from skimage import feature as ft
import cv2
from skimage import exposure
import os
import numpy as np
from numpy import *
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def HOG():
    accuracy_rate_bp=[]
    accuracy_rate_svm=[]
    x_label=[]
    for j in (2,4,6,8,10,12):
        x_label.append(str(j))
        test_X = []  # 存放测试集特征
        test_label = []  # 测试集标签
        train_X = []  # 训练集特征
        train_label = []  # 训练集标签
        # 读取测试集
        top_list = os.listdir("")
        for m in range(len(top_list)):
            img_list = os.listdir("/Users/apple/Documents/project/No4/test/" + top_list[m] + "/")
            for img_name in img_list:
                if not os.path.isdir(img_name):
                    img_path = os.path.join("/Users/apple/Documents/project/No4/test/" + top_list[m] + "/", img_name)
                    image = cv2.imread(img_path, 0)
                    # gamma_img = exposure.adjust_gamma(image, 0.8)
                    features = ft.hog(image,  # input image
                                      orientations=9,  # bin的个数
                                      pixels_per_cell=(j, j),  # cell的像素数
                                      cells_per_block=(2, 2),  # block中cell的个数
                                      block_norm='L1',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                                      transform_sqrt=True,  # power law compression (also known as gamma correction)
                                      feature_vector=True,  # flatten the final vectors
                                      visualise=False)  # return HOG map
                    test_X.append(features)
                    test_label.append(top_list[m])
                    print(img_path)
        test_X = (np.array(test_X))
        test_label = (np.array(test_label))

        # 加载训练集
        top_list = os.listdir("/Users/apple/Documents/project/No4/train/")
        for m in range(len(top_list)):
            img_list = os.listdir("/Users/apple/Documents/project/No4/train/" + top_list[m] + "/")
            for img_name in img_list:
                if not os.path.isdir(img_name):
                    img_path = os.path.join("/Users/apple/Documents/project/No4/train/" + top_list[m] + "/", img_name)
                    image = cv2.imread(img_path, 0)
                    # gamma_img = exposure.adjust_gamma(image, 0.8)
                    features = ft.hog(image,  # input image
                                      orientations=9,  # bin的个数
                                      pixels_per_cell=(j, j),  # cell的像素数
                                      cells_per_block=(2, 2),  # block中cell的个数
                                      block_norm='L1',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                                      transform_sqrt=True,  # power law compression (also known as gamma correction)
                                      feature_vector=True,  # flatten the final vectors
                                      visualise=False)  # return HOG map
                    train_X.append(features)
                    train_label.append(top_list[m])
        train_X = (np.array(train_X))
        train_label = (np.array(train_label))

        # 标准化数据
        total_X = np.vstack((test_X, train_X))
        total_X_std = (total_X - mean(total_X)) / np.std(total_X)
        test_X = total_X_std[:795]
        train_X = total_X_std[795:]

        # BP神经网络训练模型
        network = MLPClassifier(activation='tanh',  solver='adam', alpha=0.00001)
        network = network.fit(train_X, train_label)
        print("network:", network.score(test_X, test_label))
        accuracy_rate_bp.append(network.score(test_X, test_label))

        # 支持向量机
        svc = SVC(kernel='poly', degree=3, gamma=1, coef0=0)
        svc = svc.fit(train_X, train_label)
        print("svc:", svc.score(test_X, test_label))
        accuracy_rate_svm.append(svc.score(test_X,test_label))

    print(accuracy_rate_bp)
    print(accuracy_rate_svm)
    print(x_label)
    plt.ylim(0, 1)
    plt.title("HOG")
    # plt.xlabel("epochs")
    plt.xlabel("cell_size")
    plt.ylabel("accuracy_rate")
    plt.plot(x_label, accuracy_rate_bp, label="BP", marker='*', markersize=6)
    plt.plot(x_label, accuracy_rate_svm, label="SVM", marker='v', markersize=6)
    # plt.plot(ks, loss, label="loss")
    for a, b in zip(x_label, accuracy_rate_bp):
        plt.text(a, b + 0.02, '%.2f%%' % (b * 100))
    for a, b in zip(x_label, accuracy_rate_svm):
        plt.text(a, b + 0.02, '%.2f%%' % (b * 100))
    plt.legend(loc=3)
    plt.savefig(r"E:\Python\MLpractice\HOG_PCA_灰度共生矩阵\cell_size.png", dpi=1200, format='png')
    plt.pause(30)


def HOG_PCA():
    test_X = []  # 存放测试集特征
    test_label = []  # 测试集标签
    train_X = []  # 训练集特征
    train_label = []  # 训练集标签

    # 读取测试集
    top_list = os.listdir("/Users/apple/Documents/project/No4/test/")
    for m in range(len(top_list)):
        img_list = os.listdir("/Users/apple/Documents/project/No4/test/" + top_list[m] + "/")
        for img_name in img_list:
            if not os.path.isdir(img_name):
                img_path = os.path.join("/Users/apple/Documents/project/No4/test/" + top_list[m] + "/", img_name)
                image = cv2.imread(img_path, 0)
                # gamma_img = exposure.adjust_gamma(image, 0.8)
                features = ft.hog(image,  # input image
                                  orientations=10,  # bin的个数
                                  pixels_per_cell=(8, 8),  # cell的像素数
                                  cells_per_block=(2, 2),  # block中cell的个数
                                  block_norm='L1',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                                  transform_sqrt=True,  # power law compression (also known as gamma correction)
                                  feature_vector=True,  # flatten the final vectors
                                  visualise=False)  # return HOG map
                test_X.append(features)
                test_label.append(top_list[m])
                print(img_path)
    print(features.shape)
    test_X = (np.array(test_X))
    test_label = (np.array(test_label))

    # 加载训练集
    top_list = os.listdir("/Users/apple/Documents/project/No4/train/")
    for m in range(len(top_list)):
        img_list = os.listdir("/Users/apple/Documents/project/No4/train/" + top_list[m] + "/")
        for img_name in img_list:
            if not os.path.isdir(img_name):
                img_path = os.path.join("/Users/apple/Documents/project/No4/train/" + top_list[m] + "/", img_name)
                image = cv2.imread(img_path, 0)
                # gamma_img = exposure.adjust_gamma(image, 0.8)
                features = ft.hog(image,  # input image
                                  orientations=10,  # bin的个数
                                  pixels_per_cell=(8, 8),  # cell的像素数
                                  cells_per_block=(2, 2),  # block中cell的个数
                                  block_norm='L1',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                                  transform_sqrt=True,  # power law compression (also known as gamma correction)
                                  feature_vector=True,  # flatten the final vectors
                                  visualise=False)  # return HOG map
                train_X.append(features)
                train_label.append(top_list[m])
    train_X = (np.array(train_X))
    train_label = (np.array(train_label))

    # 标准化数据
    total_X = np.vstack((test_X, train_X))
    total_X_std = (total_X - mean(total_X)) / np.std(total_X)

    accuracy_rate=[]
    compoments=[]
    #PCA降维
    for j in range(1,201):
        estimator = PCA(n_components=j)
        total_X_PCA=estimator.fit_transform(total_X_std)
        test_X = total_X_PCA[:795]
        train_X = total_X_PCA[795:]

        # BP神经网络训练模型
        network = MLPClassifier()
        network = network.fit(train_X, train_label)
        print("network:", network.score(test_X, test_label),j)
        accuracy_rate.append(network.score(test_X, test_label))
        compoments.append(j)
    plt.xlim([1, 200])
    plt.ylim([0,1])
    plt.plot(compoments, accuracy_rate)
    plt.title("HOG_PCA")
    plt.draw()
    plt.pause(30)

def test():
    test_X = []  # 存放测试集特征
    test_label = []  # 测试集标签
    train_X = []  # 训练集特征
    train_label = []  # 训练集标签
    # 读取测试集
    top_list = os.listdir("/Users/apple/Documents/project/No4/test/")
    for m in range(len(top_list)):
        img_list = os.listdir("/Users/apple/Documents/project/No4/test/" + top_list[m] + "/")
        for img_name in img_list:
            if not os.path.isdir(img_name):
                img_path = os.path.join("/Users/apple/Documents/project/No4/test/"+ top_list[m] + "/", img_name)
                image = cv2.imread(img_path, 0)
                # gamma_img = exposure.adjust_gamma(image, 0.8)
                features = ft.hog(image,  # input image
                                  orientations=9,  # bin的个数
                                  pixels_per_cell=(8, 8),  # cell的像素数
                                  cells_per_block=(2, 2),  # block中cell的个数
                                  block_norm='L1',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                                  transform_sqrt=True,  # power law compression (also known as gamma correction)
                                  feature_vector=True,  # flatten the final vectors
                                  visualise=False)  # return HOG map
                test_X.append(features)
                test_label.append(top_list[m])
    test_X = (np.array(test_X))
    test_label = (np.array(test_label))

    # 加载训练集
    top_list = os.listdir("/Users/apple/Documents/project/No4/train/")
    for m in range(len(top_list)):
        img_list = os.listdir("/Users/apple/Documents/project/No4/train/" + top_list[m] + "/")
        for img_name in img_list:
            if not os.path.isdir(img_name):
                img_path = os.path.join("/Users/apple/Documents/project/No4/train/" + top_list[m] + "/", img_name)
                image = cv2.imread(img_path, 0)
                # gamma_img = exposure.adjust_gamma(image, 0.8)
                features = ft.hog(image,  # input image
                                  orientations=9,  # bin的个数
                                  pixels_per_cell=(8, 8),  # cell的像素数
                                  cells_per_block=(2, 2),  # block中cell的个数
                                  block_norm='L1',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                                  transform_sqrt=True,  # power law compression (also known as gamma correction)
                                  feature_vector=True,  # flatten the final vectors
                                  visualise=False)  # return HOG map
                train_X.append(features)
                train_label.append(top_list[m])
    train_X = (np.array(train_X))
    train_label = (np.array(train_label))

    # 标准化数据
    total_X = np.vstack((test_X, train_X))
    total_X_std = (total_X - mean(total_X)) / np.std(total_X)
    test_X = total_X_std[:795]
    train_X = total_X_std[795:]

    # BP神经网络训练模型
    network = MLPClassifier(activation='tanh', solver='adam', alpha=0.00001)
    network = network.fit(train_X, train_label)
    print("network:", network.score(test_X, test_label))


    # 支持向量机
    svc = SVC(kernel='poly', degree=3, gamma=1, coef0=0)
    svc = svc.fit(train_X, train_label)
    print("svc:", svc.score(test_X, test_label))
if __name__ == '__main__':
    HOG()
    #test()
    #HOG_PCA()
