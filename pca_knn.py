from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier

# digits = load_digits()
# train_x,test_x,train_y,test_y = train_test_split(digits.data,digits.target,test_size=0.3)
(train_image,train_lable),(test_image,test_label) = tf.keras.datasets.mnist.load_data()

X_train = train_image.reshape(60000,784)
X_test = test_image.reshape(10000,784)
y_train = train_lable
y_test = test_label

print(train_image.shape)

pca = PCA(n_components=20)
pca.fit(X_train)
pca.fit(X_test)
train_x_pca = pca.transform(X_train)
test_x_pca = pca.transform(X_test)

dt = DecisionTreeClassifier()
dt.fit(train_x_pca,y_train)
print(dt.score(test_x_pca,y_test))
# print(confusion_matrix(y_test,dt.predict(test_x_pca)))
# print(classification_report(y_test,dt.predict(test_x_pca)))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_x_pca,y_train)
print(knn.score(test_x_pca,y_test))

clf = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=200)





# #建立并训练pca模型
# pca = PCA(n_components=30)
# pca.fit(train_x)
# pca.fit(test_x)
# #返回降维后的数据集维度
# train_x_pca = pca.transform(train_x)
# test_x_pca = pca.transform(test_x)
# #降维之后的数据维度
# print(train_x_pca.shape)
#
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(train_x_pca,train_y)
# knn_predict = knn.predict(test_x_pca)
# print(classification_report(test_y,knn_predict))
# print('*'*100)
# print(confusion_matrix(test_y,knn_predict))