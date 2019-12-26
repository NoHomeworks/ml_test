from keras.datasets import mnist                #导入数据集
from keras.utils import to_categorical          #分类 one-hot编码
from keras.models import Sequential             #导入模型
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense     #导入卷积层、池化层
from keras.losses import categorical_crossentropy       #导入损失函数
from keras.optimizers import Adadelta                   #导入优化器

#数据预处理
train_X, train_y = mnist.load_data()[0]                 #获取train_X,train_y
train_X = train_X.reshape(-1, 28, 28, 1)                #对train_X数据规范化
train_X = train_X.astype('float32')
train_X /= 255
train_y = to_categorical(train_y, 10)                   #按照数字标签进行one_hot编码

#模型训练
model = Sequential()              #使用keras.model库的Sequential方法实例化模型对象
model.add(Conv2D(32, (5,5), activation='relu', input_shape=[28, 28, 1]))    #向模型添加卷积层：卷积核的个数为32，卷积核大小为5*5，激活函数为relu，图的大小为28*28，通道为1个
model.add(Conv2D(64, (5,5), activation='relu'))                             #向模型添加卷积层
model.add(MaxPool2D(pool_size=(2,2)))                       #添加最大池化层、降采样
model.add(Flatten())                                        #将模型中的数据矩阵展平
model.add(Dropout(0.5))                                     #dropout随机丢弃一些神经元，防止过拟合
model.add(Dense(128, activation='relu'))                    #添加全连接层
model.add(Dropout(0.5))                                     #随机丢弃神经元
model.add(Dense(10, activation='softmax'))                  #添加全连接层，使用relu作为激活函数

#为模型指定损失函数、优化器、评判指标
model.compile(loss=categorical_crossentropy,optimizer=Adadelta(),metrics=['accuracy'])

#训练模型
#设置批量梯度下降时的batch_size为100
#遍历所有样本的次数epoch为8
model.fit(train_X,train_y,batch_size=100,epochs=8)

#模型评估
test_X, test_y = mnist.load_data()[1]           #获取测试集数据
test_X = test_X.reshape(-1, 28, 28, 1)          #test_X数据预处理
test_X = test_X.astype('float32')
test_X /= 255
test_y = to_categorical(test_y, 10)             #对test_y进行one-hot编码
loss, accuracy = model.evaluate(test_X, test_y, verbose=1)      #使用测试集的数据进行模型评估、打印损失函数值和准确率
print('loss:%.4f accuracy:%.4f' %(loss, accuracy))
