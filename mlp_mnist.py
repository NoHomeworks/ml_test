from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

digits = load_digits()

X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.3,random_state=100)

mlp = MLPClassifier(verbose=True)
mlp.fit(X_train,y_train)
print(mlp.score(X_test,y_test))
