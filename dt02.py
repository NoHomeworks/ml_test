# from sklearn.datasets import load_digits
# import numpy as np
# import struct
# import os
# import matplotlib.pyplot as plt
#
#
# digits = load_digits()
#
#
# def load_images(file_name):
#     ##   在读取或写入一个文件之前，你必须使用 Python 内置open()函数来打开它。##
#     ##   file object = open(file_name [, access_mode][, buffering])          ##
#     ##   file_name是包含您要访问的文件名的字符串值。                         ##
#     ##   access_mode指定该文件已被打开，即读，写，追加等方式。               ##
#     ##   0表示不使用缓冲，1表示在访问一个文件时进行缓冲。                    ##
#     ##   这里rb表示只能以二进制读取的方式打开一个文件                        ##
#     binfile = open(file_name, 'rb')
#     ##   从一个打开的文件读取数据
#     buffers = binfile.read()
#     ##   读取image文件前4个整型数字
#     magic,num,rows,cols = struct.unpack_from('>IIII',buffers, 0)
#     ##   整个images数据大小为60000*28*28
#     bits = num * rows * cols
#     ##   读取images数据
#     images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
#     ##   关闭文件
#     binfile.close()
#     ##   转换为[60000,784]型数组
#     images = np.reshape(images, [num, rows * cols])
#     return images
#
# def load_labels(file_name):
#     ##   打开文件
#     binfile = open(file_name, 'rb')
#     ##   从一个打开的文件读取数据
#     buffers = binfile.read()
#     ##   读取label文件前2个整形数字，label的长度为num
#     magic,num = struct.unpack_from('>II', buffers, 0)
#     ##   读取labels数据
#     labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
#     ##   关闭文件
#     binfile.close()
#     ##   转换为一维数组
#     labels = np.reshape(labels, [num])
#     return labels
#
# train_images = load_images(filename_train_images)
# train_labels = load_labels(filename_train_labels)
# text_images = load_images(filename_text_images)
# text_labels = load_labels(filename_text_labels)
#
# print(train_labels[0])
# print(text_labels[0])

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

train_x = digits.data
train_y = digits.target


# plt.imshow(digits.images[0],camp = plt.cm.gray_r,interpolation='nearest')
# plt.imshow(digits.images[0],cmap = plt.cm.gray_r,interpolation = 'nearest')

import numpy as np
from sklearn import tree

clf = tree.DecisionTreeClassifier()
from sklearn import datasets
digits = datasets.load_digits()

import matplotlib.pyplot as plt

plt.subplot(321)
plt.imshow(digits.images[1791],cmap = plt.cm.gray_r,interpolation = 'nearest')
plt.subplot(322)
plt.imshow(digits.images[1792],cmap = plt.cm.gray_r,interpolation = 'nearest')
plt.subplot(323)
plt.imshow(digits.images[1793],cmap = plt.cm.gray_r,interpolation = 'nearest')
plt.subplot(324)
plt.imshow(digits.images[1794],cmap = plt.cm.gray_r,interpolation = 'nearest')
plt.subplot(325)
plt.imshow(digits.images[1795],cmap = plt.cm.gray_r,interpolation = 'nearest')
plt.subplot(326)
plt.imshow(digits.images[1796],cmap = plt.cm.gray_r,interpolation = 'nearest')

clf.fit(digits.data[1:1617], digits.target[1:1617])
predict = clf.predict(digits.data[1618:])

accurancy = np.sum(np.equal(predict,digits.target[1618:]))/180

print(accurancy)
print("实际数字：",digits.target[1791:])
print("预测数字：",clf.predict(digits.data[1791:]))


