import tensorflow as tf
from sklearn.decomposition import PCA
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.mnist.load_data()

for i in range(20):
    plt.imshow(train_images[i])
    plt.show()


# train_x = train_images.reshape(60000,784)
# test_x = test_images.reshape(10000,784)
# train_y = train_labels
# test_y = test_labels

# demisions = range(20,40,10)
#
#
# score = []
# for demision in demisions:
#     pca = PCA(n_components=demision)
#     data = np.vstack((train_x,test_x))
#     train_x = data[:60000]
#     test_x = data[60000:]
#     train_x_pca = pca.fit_transform(train_x)
#     test_x_pca = pca.fit_transform(test_x)
#     clf = DecisionTreeClassifier()
#     clf.fit(train_x_pca,train_y)
#     score.append(clf.score(test_x_pca,test_y))
#
# print(score)

# data = np.vstack((train_x,test_x))
# train_x = data[:60000]
# test_x = data[60000:]
# pca = PCA(n_components=400)
# train_x_pca = pca.fit_transform(train_x)
# test_x_pca = pca.fit_transform(test_x)
#
# clf = KNeighborsClassifier()
# clf.fit(train_x_pca,train_y)
# print(clf.score(test_x_pca,test_y))
