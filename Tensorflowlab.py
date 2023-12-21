from __future__ import absolute_import, division, print_function, unicode_literals
#set TF_MIN_GPU_MULTIPROCESSOR_COUNT=4
import tensorflow as tf
import numpy as np
from tensorflow import keras as ks
import os
tf.debugging.set_log_device_placement(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
(training_images, training_labels),(test_images, test_labels)=ks.datasets.mnist.load_data()
print('Training Images Dataset Shape: {}'.format(training_images.shape))
training_images = training_images/255.0
test_images = test_images/255.0
training_images = training_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
hidden_1_activation_function = 'relu'
hidden_2_activation_function = 'sigmoid'
output_activation_function = 'sigmoid'
optimizer = 'adam'
loss_function = 'CategoricalCrossentropy'
metric = ['accuracy']
tf_callback = tf.keras.callbacks.TensorBoard(log_dir="./mylogdir")
cnn_model = tf.keras.models.Sequential()
cnn_model.add(ks.layers.Conv2D(30, (20, 20), activation='relu', input_shape=(28, 28, 1), name='Conv2D_layer'))
cnn_model.add(ks.layers.MaxPooling2D((2, 2), name='Maxpooling_2D'))
cnn_model.add(ks.layers.Flatten(name='Flatten'))
cnn_model.add(ks.layers.Dense(50, activation='relu', name='Hidden_layer'))
cnn_model.add(ks.layers.Dense(10, activation='softmax', name='Output_layer'))
cnn_model.summary()
with tf.device('/device:GPU:1'):
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(training_images, training_labels , epochs=40, callbacks=[tf_callback])