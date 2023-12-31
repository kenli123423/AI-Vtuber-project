import tensorflow as tf
import keras
#creating a simple densely-connected layer
import pathlib

import tensorflow.python.keras.models

config = {
    'batch_size':128,
    'learning_rate':0.001,
    'output_layer_activation':'softmax',
    'optimizer':'SGD',
    'dense_layers':[
        {'neurons':128, 'activation':'relu'},
        {'neurons':64, 'activation':'relu'},
    ]
}
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
class_names = train_ds.class_names
num_classes = len(class_names)
tf_callback = tf.keras.callbacks.TensorBoard(log_dir="../mylogdir")
class Model(tensorflow.keras.Model):
    def __init__(self, output_size):
        super(Model, self).__init__()
        self.Rescaling = tf.keras.layers.Rescaling(1./255, input_shape=(180,180,3))
        self.Conv2D_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(2,2), activation='relu', padding='valid')
        self.MaxPooling2D = tf.keras.layers.MaxPooling2D(pool_size=(3,3))
        self.Conv2D_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same', dilation_rate=1)
        self.MaxPooling2D_1 = tf.keras.layers.MaxPooling2D(pool_size=(4,4))
        self.Conv2D_3 = tf.keras.layers.Conv2D(filters=96, kernel_size=(4,4), padding='same', dilation_rate=1)
        self.Flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.out = tf.keras.layers.Dense(units=output_size, activation='softmax')

    def call(self, inputs, training=True, **kwargs):
        x = self.Rescaling(inputs)
        x = self.Conv2D_1(x)
        x = self.MaxPooling2D(x)
        x = self.Conv2D_2(x)
        x = self.MaxPooling2D_1(x)
        x = self.Conv2D_3(x)
        x = self.Flatten(x)
        x = self.dense_1(x)
        return self.out(x)
s = Model(output_size=num_classes)
s.compile(
    optimizer=tf.optimizers.Adam(),
    loss = tf.keras.losses.sparse_categorical_crossentropy,
    metrics = ['accuracy']
)
with tf.device('/device:GPU:1'):
    s.fit(train_ds, epochs=20, batch_size=64)





