import tensorflow as tf
from sklearn.decomposition import PCA
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

(train_x,train_y),(test_x,test_y) = tf.keras.datasets.mnist.load_data()

print(train_x.shape)

train_x = train_x.reshape(60000,784)
test_x = test_x.reshape(10000,784)


score = []
for i in range(20,200,10):
    pca = PCA(n_components=i)
    data  = np.vstack((train_x,test_x))
    train_x = data[:60000]
    test_x = data[60000:]
    train_x_pca = pca.fit_transform(train_x)
    test_x_pca = pca.fit_transform(test_x)
    clf = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=200)
    clf.fit(train_x_pca,train_y)
    score.append(clf.score(test_x_pca,test_y))


plt.plot(range(20,200,10),score,marker='*',c='r')
plt.xlabel('dimension')
plt.ylabel('accuracy')
plt.title('')
plt.show()