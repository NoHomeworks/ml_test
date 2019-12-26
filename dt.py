from sklearn import tree
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot = False)
train_num = 60000
test_num = 10000

#获取训练数据与测试数据
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

if __name__ == '__main__':
    #构建决策树分类器
    clf = tree.DecisionTreeClassifier()
    #使用训练数据放入分类器中训练
    clf.fit(x_train[:train_num],y_train[:train_num])
    #使用测试数据进行预测
    prediction = clf.predict(x_test[:test_num])
    #计算预测结果与真实结果的准确性
    accurancy = np.sum(np.equal(prediction,y_test[:test_num])) / test_num

    print('accurancy:',accurancy)