import tensorflow as tf

(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape(train_images.shape[0],28,28,1)
test_images = test_images.reshape(test_images.shape[0],28,28,1)

def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(6,input_shape= (28,28,1),kernel_size = (5,5),strides = 1,padding = 'same',activation = tf.nn.relu))
    model.add(tf.keras.layers.MaxPooling2D(pool_size =(2,2),strides = 2))
    model.add(tf.keras.layers.Conv2D(10,kernel_size = (3,3),strides = 1,activation = tf.nn.relu))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2),strides = 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation("softmax"))


    model.compile(loss = "sparse_categorical_crossentropy",optimizer = 'Adadelta',metrics = ['accuracy'])
    return model



model = create_model()
history = model.fit(train_images,train_labels,epochs = 15)