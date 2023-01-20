from tensorflow.keras.applications import ResNet50, EfficientNetB0, ResNet101, EfficientNetB3
import tensorflow as tf
import tensorflow_datasets as tfds

batch_size = 64

dataset_name = sys.argv[1].lower()
(ds_train, ds_test), ds_info = tfds.load(
        dataset_name, split=["train", "test"], with_info=True, as_supervised=True
)
NUM_CLASSES = ds_info.features["label"].num_classes

IMG_SIZE = 224

size = (IMG_SIZE, IMG_SIZE)
ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

# One-hot / categorical encoding
def input_preprocess(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


ds_train = ds_train.map(
    input_preprocess, num_parallel_calls=tf.data.AUTOTUNE
)
ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(input_preprocess)
ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)

from tensorflow.keras.applications import EfficientNetB0, ResNet50, ResNet101, DenseNet121

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
#x = img_augmentation(inputs)
outputs = DenseNet121(include_top=True, classifier_activation=None, weights=None, classes=NUM_CLASSES)(inputs)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
)

model.summary()

epochs = 80  # @param {type: "slider", min:10, max:100}
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=1)

model.save("trained_model.h5")
