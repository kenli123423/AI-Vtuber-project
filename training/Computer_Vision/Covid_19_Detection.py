import tensorflow as tf
from tensorflow import keras
import os
#set TF_MIN_GPU_MULTIPROCESSOR_COUNT=4
from keras import mixed_precision
optimizer = keras.optimizers.Adam()
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
path = "F:\PyCharm Community Edition 2022.2.1\Data\COVID-19_Radiography_Dataset"
train_ds = keras.preprocessing.image_dataset_from_directory(
    path,
    image_size=(200,200),
    seed=3,
    subset = 'training',
    validation_split=0.1,
    batch_size=128
)
class_names = train_ds.class_names
num_class = len(class_names)
print(class_names)

class Model(tf.keras.Model):
    def __init__(self, output_size):
        super(Model, self).__init__()
        self.conv_1 = keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(250,250,3), dilation_rate=1)
        self.pooling_1 = keras.layers.MaxPooling2D(pool_size=(3,3), padding='valid')
        self.conv_2  = keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')
        self.pooling_2 = keras.layers.MaxPooling2D(pool_size=(8,8), padding='same')
        self.conv_3 = keras.layers.Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu', dilation_rate=1)
        self.pooling_3 = keras.layers.MaxPooling2D(pool_size=(5,5))

        self.Flatten = keras.layers.Flatten()
        self.dense_1 = keras.layers.Dense(512,activation='relu')
        self.normalisation = keras.layers.BatchNormalization()
        self.dense_2 = keras.layers.Dense(512, activation='relu')
        self.Dropout_1 = keras.layers.Dropout(0.1)
        self.normalisation_1 = keras.layers.BatchNormalization()
        self.dense_3 = keras.layers.Dense(512, activation='relu')
        self.normalisation_2 = keras.layers.BatchNormalization()
        self.dense_4 = keras.layers.Dense(num_class, activation='softmax')

    def call(self, inputs, trainable=True, **kwargs):
        x = self.conv_1(inputs)
        x = self.pooling_1(x)
        x = self.conv_2(x)
        x = self.pooling_2(x)
        x = self.conv_3(x)
        x = self.pooling_3(x)
        x = self.Flatten(x)
        x = self.dense_1(x)
        x = self.normalisation(x)
        x = self.dense_2(x)
        x = self.normalisation_1(x)
        x = self.dense_3(x)
        x = self.normalisation_2(x)
        return self.dense_4(x)
s = Model(num_class)
s.compile(
    optimizer=optimizer,
    loss=keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)
with tf.device('/device:GPU:1'):
    s.fit(train_ds, epochs=1)
    s.save("F:\PyCharm Community Edition 2022.2.1\Checkpoint")