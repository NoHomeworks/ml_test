from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

digits = load_digits()

X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.3,random_state=100)

#高斯贝叶斯
gaussian = GaussianNB()
gaussian.fit(X_train,y_train)
print(gaussian.score(X_test,y_test))

#多项式分布贝叶斯
multi = MultinomialNB()
multi.fit(X_train,y_train)
print(multi.score(X_test,y_test))

#伯努利贝叶斯
bernouli = BernoulliNB()
bernouli.fit(X_train,y_train)
print(bernouli.score(X_test,y_test))