import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


(train_images,train_labels),(test_images,test_labels)=tf.keras.datasets.mnist.load_data()

train_X = train_images.reshape(60000,784)
test_X = test_images.reshape(10000,784)
train_y = train_labels
test_y = test_labels

for i in range(1,21):
    plt.subplot(4,5,i)
    plt.imshow(train_images[i])
plt.show()


# #建立决策树分类器模型
# dt = DecisionTreeClassifier()
# #训练分类器
# dt.fit(train_X,train_y)
# #预测结果
# dt_predict = dt.predict(test_X)
# #评价结果
# print('决策树分类器的预测准确率为：',dt.score(test_X,test_y))
# print('决策树分类器的混淆矩阵为：\n',confusion_matrix(dt_predict,test_y))
# print('决策树分类器的分类报告为：\n',classification_report(dt_predict,test_y))
# print('*'*50)

#建立knn分类器模型
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(train_X,train_y)
# knn_predict = knn.predict(test_X)
# print('knn分类器的预测准确率为：',knn.score(test_X,test_y))
# print('knn分类器的混淆矩阵为：\n',confusion_matrix(knn_predict,test_y))
# print('knn分类器的分类报告为：\n',classification_report(knn_predict,test_y))

# #建立高斯贝叶斯分类器模型
# gaussian = GaussianNB()
# gaussian.fit(train_X,train_y)
# gaussian_predict = gaussian.predict(test_X)
# print('高斯贝叶斯分类器的预测准确率为：',gaussian.score(test_X,test_y))
# print('高斯贝叶斯分类器的混淆矩阵为：',confusion_matrix(gaussian_predict,test_y))
# print('高斯贝叶斯分类器的分类报告为：',classification_report(gaussian_predict,test_y))
#
# #建立多项式贝叶斯分类器模型
# multi = MultinomialNB()
# multi.fit(train_X,train_y)
# multi_predict = multi.predict(test_X)
# print('多项式贝叶斯分类器的预测准确率为：',multi.score(test_X,test_y))
# print('多项式贝叶斯分类器的混淆矩阵为：',confusion_matrix(multi_predict,test_y))
# print('多项式贝叶斯分类器的分类报告为：',classification_report(multi_predict,test_y))
#
# #建立伯努利贝叶斯分类器
# bernoulli = BernoulliNB()
# bernoulli.fit(train_X,train_y)
# bernoulli_predict = bernoulli.predict(test_X)
# print('伯努利贝叶斯分类器的预测准确率为：',bernoulli.score(test_X,test_y))
# print('伯努利贝叶斯分类器的混淆矩阵为：',confusion_matrix(bernoulli_predict,test_y))
# print('伯努利贝叶斯分类器的分类报告为：',classification_report(bernoulli_predict,test_y))

#建立多层感知器模型
# mlp = MLPClassifier()
# mlp.fit(train_X,train_y)
# mlp_predict = mlp.predict(test_X)
# print('多层感知器的预测准确率为：',mlp.score(test_X,test_y))
# print('多层感知器的混淆矩阵为：',confusion_matrix(mlp_predict,test_y))
# print('多层感知器的分类报告为：',classification_report(mlp_predict,test_y))

#集成学习——AdabBoost
# abc = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=200)
# abc.fit(train_X,train_y)
# abc_predict = abc.predict(test_X)
# print('AdaBoost的预测准确率为：',abc.score(test_X,test_y))
# print('AdaBoost的混淆矩阵为：',confusion_matrix(abc_predict,test_y))
# print('AdaBoost的分类报告为：',classification_report(abc_predict,test_y))

#集成学习——Bagging
# bag = BaggingClassifier(KNeighborsClassifier(),n_estimators=200)
# bag.fit(train_X,train_y)
# bag_predit = bag.predict(test_X)
# print('Bagging的预测准确率为：',bag.score(test_X,test_y))
# print('Bagging的混淆矩阵为：',confusion_matrix(bag_predit,test_y))
# print('Bagging的分类报告为：',classification_report(bag_predit,test_y))
