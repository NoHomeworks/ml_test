import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
digits = load_digits()

#分割数据集
X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.3,random_state=42)

#设置参数矩阵
param_grid = [{'criterion':['entropy'],"min_impurity_decrease": np.linspace(0,1,100)},
              {'criterion':['gini'],'min_impurity_decrease':np.linspace(0,0.2,100)},
              {'max_depth':np.arange(2,30,2)},
              {'min_samples_split':np.arange(2,30,2)}]
# #构建决策树
clf = GridSearchCV(DecisionTreeClassifier(),param_grid,cv=5)
#训练决策树模型
clf.fit(X_train,y_train)
#预测决策树模型
clf.predict(X_test)
#模型评估
print('决策树分类器最佳参数组合为：',clf.best_params_)
print('在此参数组合下的准确率为：',clf.best_score_)

#设置参数矩阵
param_grid = {'n_neighbors':np.arange(1,10,2)}

clf = GridSearchCV(KNeighborsClassifier(),param_grid,cv=5)
clf.fit(X_train,y_train)
clf.predict(X_test)
print('knn分类器最佳参数组合为：',clf.best_params_)
print('在此参数组合下的准确率为：',clf.best_score_)