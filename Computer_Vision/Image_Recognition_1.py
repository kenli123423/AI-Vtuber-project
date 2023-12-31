import tensorflow as tf
from tensorflow import keras
import keras as ks
import os
import pathlib
dataset_url = "https://storage.googleapis.com/download.\
tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(180,180),
    batch_size=32
)
#Data Splitting
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(180,180),
    batch_size=16
)
tf_callback = tf.keras.callbacks.TensorBoard(log_dir="../mylogdir")
class_names = train_ds.class_names
num_classes = len(class_names)
nn_model = tf.keras.models.Sequential()
nn_model.add(keras.layers.Rescaling(1./255, input_shape=(180, 180 ,3)))
nn_model.add(keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid', activation='relu'))
nn_model.add(keras.layers.MaxPooling2D())
nn_model.add(keras.layers.Conv2D(filters=32, kernel_size=(2,2), padding='same', activation='relu'))
nn_model.add(keras.layers.MaxPooling2D())
nn_model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
nn_model.add(keras.layers.Flatten())
nn_model.add(keras.layers.Dense(units=128, activation='relu'))
nn_model.add(keras.layers.Dense(num_classes, activation='softmax'))
nn_model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
with tf.device('/device:GPU:1'):
    nn_model.fit(train_ds, epochs=10)
