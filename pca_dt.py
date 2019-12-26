from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

digits = load_digits()

train_x,test_x,train_y,test_y = train_test_split(digits.data,digits.target,test_size=0.2)

# pca = PCA(n_components=20)
# pca.fit(train_x)
# pca.fit(test_x)
#
# train_x_pca = pca.transform(train_x)
# test_x_pca = pca.transform(test_x)
#
# clf = DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=50)
#
# clf.fit(train_x_pca,train_y)
# print(clf.score(train_x_pca,train_y))
# print(clf.score(test_x_pca,test_y))
#
train_score = []
test_score = []

for i in range(1,64):
    pca = PCA(n_components=i)
    pca.fit(train_x)
    pca.fit(test_x)

    train_x_pca = pca.transform(train_x)
    test_x_pca = pca.transform(test_x)

    clf = DecisionTreeClassifier()
    clf.fit(train_x_pca,train_y)
    train_score.append(clf.score(train_x_pca,train_y))
    test_score.append(clf.score(test_x_pca,test_y))

plt.plot(range(1,64),train_score,marker = '*',c='r',label = 'train set')
plt.plot(range(1,64),test_score,marker = '+',c='b',label = 'test set')
plt.title('hello')
plt.xlabel('n_components')
plt.ylabel('score')
plt.legend(loc='best')
plt.show()