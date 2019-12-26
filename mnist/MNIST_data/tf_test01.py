import tensorflow as tf

(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images / 255.0
test_labels = test_labels / 255.0
train_images = train_images.reshape(train_images.shape[0],28,28,1)
test_images = test_images.reshape(test_images.shape[0],28,28,1)

def create_model():
    model = tf.keras.Sequential()
    model.add()