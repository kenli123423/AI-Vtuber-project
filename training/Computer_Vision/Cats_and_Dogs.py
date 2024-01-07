import tensorflow as tf
import numpy as np
import keras
import matplotlib as plt
from tensorflow import keras
from keras import Sequential
from keras.applications import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras.layers import Dense,Flatten, Input, Dropout
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import pathlib
path_cats = "F:\PyCharm Community Edition 2022.2.1\\Data\\archive (2)\\test_set\\test_set\cats"
path_dogs = "F:\PyCharm Community Edition 2022.2.1\Data\\archive (2)\test_set\\test_set\dogs"
cat_dir = os.path.join(path_cats)
dog_dir = os.path.join(path_dogs)
base_dir = "F:\PyCharm Community Edition 2022.2.1\\Data\\archive (2)\\test_set\\test_set"
base_dir1 = "F:\PyCharm Community Edition 2022.2.1\\Data\\archive (2)\\training_set\\training_set"
optimizer = keras.optimizers.Adam()
train_datagen = keras.preprocessing.image_dataset_from_directory(
    base_dir,
    image_size=(300,300),
    subset='training',
    seed = 1,
    validation_split=0.1,
    batch_size=32
)
test_datagen = keras.preprocessing.image_dataset_from_directory(
    base_dir,
    image_size=(200,200),
    subset='validation',
    seed=1,
    validation_split=0.1,
    batch_size=32
)
class_name = train_datagen.class_names
class Model(tf.keras.Model):
    def __init__(self, output_size):
        super(Model, self).__init__()
        self.aug = keras.layers.RandomFlip(mode='horizontal_and_vertical')
        self.rot = keras.layers.RandomRotation(0.2)
        self.conv = keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')
        self.pooling = keras.layers.MaxPooling2D()
        self.conv_1 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(300,300,3))
        self.pooling_1 = keras.layers.MaxPooling2D(pool_size=(3,3), padding='valid')
        self.conv_2  = keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')
        self.pooling_2 = keras.layers.MaxPooling2D(pool_size=(4,4), padding='valid')
        self.conv_3 = keras.layers.Conv2D(filters=64, kernel_size=(4,4), padding='same')
        self.pooling_3 = keras.layers.MaxPooling2D(pool_size=(5,5),strides=1)

        self.Flatten = keras.layers.Flatten()
        self.dense_1 = keras.layers.Dense(512,activation='relu')
        self.normalisation = keras.layers.BatchNormalization()
        self.dense_2 = keras.layers.Dense(512, activation='relu', use_bias=False)
        self.Dropout_1 = keras.layers.Dropout(0.1)
        self.normalisation_1 = keras.layers.BatchNormalization()
        self.dense_3 = keras.layers.Dense(512, activation='relu')
        self.normalisation_2 = keras.layers.BatchNormalization()
        self.dense_4 = keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, trainable=True, **kwargs):
        x = self.aug(inputs)
        x = self.rot(x)
        x = self.conv(x)
        x = self.pooling(x)
        x = self.conv_1(x)
        x = self.pooling_1(x)
        x = self.conv_2(x)
        x = self.pooling_2(x)
        x = self.conv_3(x)
        x = self.pooling_3(x)
        x = self.Flatten(x)
        x = self.dense_1(x)
        x = self.normalisation(x)
        x = self.dense_2(x)
        x = self.Dropout_1(x)
        x = self.normalisation_1(x)
        x = self.dense_3(x)
        x = self.normalisation_2(x)
        return self.dense_4(x)

s=Model(output_size=len(class_name))
s.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)
with tf.device('/device:GPU:1'):
    s.fit(train_datagen, epochs=10)