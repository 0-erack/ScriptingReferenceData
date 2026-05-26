import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

dataset_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = dataset_mnist.load_data()
class_types = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0
plt.figure(figsize=(10,5))
for i in range(15):
    plt.subplot(3,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_types[train_labels[i]])
plt.show()

model_1 = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)), keras.layers.Dense(128, activation=tf.nn.relu), keras.layers.Dense(10, activation=tf.nn.softmax)])
model_1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_2 = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)), keras.layers.Dense(128, activation=tf.nn.relu), keras.layers.Dense(10, activation=tf.nn.softmax)])
model_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_1.fit(train_images, train_labels, validation_split=0.4, epochs=2, verbose=2)
model_2.fit(train_images, train_labels, validation_split=0.4, epochs=15, verbose=2)
score_1 = model_1.evaluate(test_images, test_labels, verbose=0)
score_2 = model_2.evaluate(test_images, test_labels, verbose=0)
print(score_1)
print(score_2)