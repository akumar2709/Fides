from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
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
import sys
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0, DenseNet201, EfficientNetB7, MobileNetV2, DenseNet121
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

IMG_SIZE = 224

batch_size = 32

dataset_name = sys.argv[3].lower()
(ds_train, ds_test), ds_info = tfds.load(
    dataset_name, split=["train", "test"], with_info=True, as_supervised=True
)
NUM_CLASSES = ds_info.features["label"].num_classes

size = (IMG_SIZE, IMG_SIZE)
ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))

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

student = sys.argv[1].lower()
teacher = sys.argv[2].lower()

#teacher = tf.keras.models.load_model("ResNet152_cifar10_nsm.h5")
#student = tf.keras.models.load_model("ResNet50_cifar10_nsm.h5")

student.trainable = False

#Unfreezing layers in models
def unfreeze_model(model, layers):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

unfreeze_model(student, -20)


train_loss = tf.keras.metrics.Mean(name="train_loss")
valid_loss = tf.keras.metrics.Mean(name="test_loss")

train_acc = tf.keras.metrics.CategoricalAccuracy(name="train_acc")
valid_acc = tf.keras.metrics.CategoricalAccuracy(name="valid_acc")

#Knowledge Distillation Loss
def get_kd_loss(student_logits, teacher_logits,
                true_labels, temperature,
                alpha, beta):
    teacher_probs = tf.nn.softmax(teacher_logits / temperature)
    kd_loss = tf.keras.losses.categorical_crossentropy(
        teacher_probs, student_logits / temperature,
        from_logits=True)

    ce_loss = tf.keras.losses.categorical_crossentropy(
        true_labels, student_logits, from_logits=True)

    total_loss = (alpha * kd_loss) + (beta * ce_loss)
    return total_loss

class Student(tf.keras.Model):
    def __init__(self, trained_teacher, student,
                 temperature=5, alpha=0.8, beta=0.2):
        super(Student, self).__init__()
        self.trained_teacher = trained_teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def train_step(self, data):
        images, labels = data
        teacher_logits = self.trained_teacher(images)

        with tf.GradientTape() as tape:
            student_logits = self.student(images)
            loss = get_kd_loss(student_logits, teacher_logits,
                               labels, self.temperature,
                               self.alpha, self.beta)
        gradients = tape.gradient(loss, self.student.trainable_variables)
        # As mentioned in Section 2 of https://arxiv.org/abs/1503.02531
        gradients = [gradient * (self.temperature ** 2) for gradient in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        train_loss.update_state(loss)
        train_acc.update_state(labels, tf.nn.softmax(student_logits))
        t_loss, t_acc = train_loss.result(), train_acc.result()
        train_loss.reset_states(), train_acc.reset_states()
        return {"train_loss": t_loss, "train_accuracy": t_acc}

    
    def test_step(self, data):
        images,labels = data
        teacher_logits = self.trained_teacher(images)

        student_logits = self.student(images, training=False)
        loss = get_kd_loss(student_logits, teacher_logits,
                               labels, self.temperature,
                               self.alpha, self.beta)

        valid_loss.update_state(loss)
        valid_acc.update_state(labels, tf.nn.softmax(student_logits))
        v_loss, v_acc = valid_loss.result(), valid_acc.result()
        valid_loss.reset_states(), valid_acc.reset_states()
        return {"loss": v_loss, "accuracy": v_acc}

student_distilled = Student(teacher, student)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
student_distilled.compile(optimizer)
student_distilled.student.compile(optimizer)
student_distilled.fit(ds_train, 
            validation_data= ds_test,
            epochs=10)

student_distilled.student.save("distilled_model.h5")
