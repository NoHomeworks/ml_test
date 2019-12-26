import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


(train_image,train_lable),(test_image,test_label) = tf.keras.datasets.mnist.load_data()

train_label_onehot = tf.keras.utils.to_categorical(train_lable)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28,28)))
model.add(tf.keras.layers.Dense(16,activation = "sigmoid"))
model.add(tf.keras.layers.Dense(16,activation = "sigmoid"))
model.add(tf.keras.layers.Dense(10,activation = "softmax"))

model.compile(optimizer = tf.keras.optimizers.Adam(lr=1e-4),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

from time import time
startTime = time()
test_lable_onehot = tf.keras.utils.to_categorical(test_label)
history = model.fit(train_image,train_label_onehot,batch_size = 100,epochs = 100,validation_data = (test_image,test_lable_onehot))
duration = time() - startTime
print("duration",duration)

