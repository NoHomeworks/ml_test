#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 16:50:11 2019

@author: apple
"""

"""
sklearn中的load_digits数据集简介：
手写字符体数据集,有['images', 'data', 'target_names', 'DESCR', 'target']这5个属性，

images:ndarray类型，保存8*8的图像，一共1797张图片。
data:将images展开成一行，一共有1797行，64（8*8）列，相当于属性x
target:每张图片代表的数字，相当于y
target:数据集中所有的标签，即0-9
DESCR:数据集的描述，包括作者，时间，数据来源等等。

"""

"""
1.读数据
2.数据预处理
3.分割数据集
4.建模
5.训练
6.预测
7.评价
"""


import numpy as np
from sklearn import tree
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import LinearSVC


#读入数据集
digits = load_digits()

#数据集分割
X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.3,random_state=42)

#构建决策树分类器
dt = tree.DecisionTreeClassifier(criterion="entropy",#用熵计算(也可换成基尼系数)
                                 splitter="best",#best适合样本量不大，random适合样本量大
                                 max_depth=10,#最大深度，通常用于解决过拟合
                                 min_samples_leaf=5)#解决过拟合
#对训练集进行训练
dt.fit(X_train,y_train)
#对测试集预测
dt_predict = dt.predict(X_test)
#评价分类器性能
print("决策树分类器的准确率为：",dt.score(X_test,y_test))
#输出混淆矩阵
print('决策树分类器的混淆矩阵为：\n',confusion_matrix(dt_predict,y_test))
#输出评价报告
print('决策树分类器的分类报告为:\n',classification_report(dt_predict,y_test))
print('决策树分类器的参数为：',dt.get_params())
print('*'*200)

#构建knn分类器
neigh = KNeighborsClassifier(n_neighbors = 1) #选取最近的3个邻居
neigh.fit(X_train,y_train)
neigh_predict = neigh.predict(X_test)
print('knn分类器的准确率为：',neigh.score(X_test,y_test))
print('knn分类器的混淆矩阵为：\n',confusion_matrix(neigh_predict,y_test))
print('knn分类器的分类报告为：\n',classification_report(dt_predict,y_test))
print('knn分类器的参数为：',neigh.get_params())
print('*'*200)

#构建高斯朴素贝叶斯分类器
bayes = GaussianNB(priors=None)
bayes.fit(X_train,y_train)
bayes_predict = bayes.predict(X_test)
print("高斯Bayes分类器的准确率为:",bayes.score(X_test,y_test))
print('高斯Bayes分类器的混淆矩阵为：\n',confusion_matrix(bayes_predict,y_test))
print('高斯Bayes分类器的分类报告为：\n',classification_report(bayes_predict,y_test))
print('高斯Bayes分类器的参数为：',bayes.get_params())
print('*'*200)

#构建多项式分布贝叶斯分类器
multi = MultinomialNB()
multi.fit(X_train,y_train)
multi_predict = multi.predict(X_test)
print('多项式分布Bayes分类器的准确率为：',multi.score(X_test,y_test))
print('多项式分布Bayes分类器的混淆矩阵为：\n',confusion_matrix(multi_predict,y_test))
print('多项式分布Bayes分类器的分类报告为：\n',classification_report(multi_predict,y_test))
print('多项式分布Bayes分类器的参数为：',multi.get_params())
print('*'*200)

#构建伯努利贝叶斯分类器
bernouli = BernoulliNB()
bernouli.fit(X_train,y_train)
bernouli_predict = bernouli.predict(X_test)
print('伯努利Bayes分类器的准确率为：',bernouli.score(X_test,y_test))
print('伯努利Bayes分类器的混淆矩阵为：\n',confusion_matrix(bernouli_predict,y_test))
print('伯努利Bayes分类器的分类报告为：\n',classification_report(bernouli_predict,y_test))
print('伯努利Bayes分类器的参数为：',bernouli.get_params())
print('*'*200)

#构建多层感知器分类器
mlp = MLPClassifier()
mlp.fit(X_train,y_train)
mlp_predict = mlp.predict(X_test)
print("多层感知器分类器的准确率为",mlp.score(X_test,y_test))
print('多层感知器分类器的混淆矩阵为：\n',confusion_matrix(mlp_predict,y_test))
print('多层感知器分类器的分类报告为：\n',classification_report(mlp_predict,y_test))
print('多层感知器分类器的参数为：',mlp.get_params())
print('*'*200)

#构建集成学习——AdaBoost分类器
abc = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(),n_estimators=200)#生成200颗决策树
abc.fit(X_train,y_train)
abc_predict = abc.predict(X_test)
print('AdaBoost分类器的准确率为',abc.score(X_test,y_test))
print('AdaBoost分类器的混淆矩阵为：\n',confusion_matrix(abc_predict,y_test))
print('AdaBoost分类器的分类报告为：\n',classification_report(abc_predict,y_test))
print('*'*200)

#构建集成学习——Bagging分类器
bag = BaggingClassifier(KNeighborsClassifier(),n_estimators=200,max_samples=0.5,max_features=0.5)
bag.fit(X_train,y_train)
bag_predict = bag.predict(X_test)
print('Bagging分类器的准确率为',bag.score(X_test,y_test))
print('Bagging分类器的混淆矩阵为：\n',confusion_matrix(bag_predict,y_test))
print('Bagging分类器的分类报告为：\n',classification_report(bag_predict,y_test))


print("knn分类器预测数字：",neigh.predict(digits.data[1777:]))
print("实际数字：",digits.target[1777:])








# #将20张图片可视化
# for i in range(1,21):
#     plt.subplot(4,5,i)
#     plt.imshow(digits.images[1776+i])
# plt.show()