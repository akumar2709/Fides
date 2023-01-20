import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf
from keras.utils import np_utils
from keras.models import load_model
from keras.datasets import cifar100, cifar10
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

detection_model = models.Sequential()
detection_model.add(keras.Input(shape=(1)))
detection_model.add(layers.Dense(128, activation='LeakyReLU'))
detection_model.add(layers.Dropout(0.2))
detection_model.add(layers.BatchNormalization())
detection_model.add(layers.Dense(256, activation='LeakyReLU'))
detection_model.add(layers.Dropout(0.2))
detection_model.add(layers.BatchNormalization())
detection_model.add(layers.Dense(128, activation='LeakyReLU'))
detection_model.add(layers.Dropout(0.2))
detection_model.add(layers.BatchNormalization())
detection_model.add(layers.Dense(1))

loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) # Low since we are fine-tuning

detection_model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])


converter = tf.lite.TFLiteConverter.from_keras_model(detection_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_ops = [
#  tf.lite.OpsSet.TFLITE_BUILTINS # enable TensorFlow Lite ops.
#  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
#]
#converter.target_spec.supported_types = [tf.float32]
tflite_model = converter.convert()
#tflite_quant_model = converter.convert()

with open('attack_detection_benchmark.tflite', 'wb') as f:
  f.write(tflite_model)
