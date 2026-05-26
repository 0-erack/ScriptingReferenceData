import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import tensorflow as tf
import tensorflow_datasets as tfds

digits = sklearn.datasets.load_digits()
figure = plt.figure(figsize=(12,6))
for i in range(1,11):
    image = np.array(digits['images'][i], dtype="float")
    figure.add_subplot(2,5,i)
    plt.imshow(image, cmap='Greys')
plt.show()

print(digits)
print(str(digits['target_names'].shape))
print(str(digits['target'].shape))
print(str(digits['images'].shape))

'''X, y = sklearn.datasets.load_boston(return_X_y=True)
fiugre = plt.figure(figsize=(16,16))
for i in range(X.shape[1]):
    ax = figure.add_sublot(4,4,i+1)
    ax.set_ylabel('price')
    ax.set_xlabel('feature' + str(i+1))
    plt.scatter(X[:,i], y)
plt.show()'''

tfds.list_builders()
tfds.disable_progress_bar()
my_data = tfds.load(name='iris', batch_size=-1)
numpy_ds = tfds.as_numpy(my_data)
numpy_data, numpy_labels = numpy_ds['train']['features'], numpy_ds['train']['label']

figure = plt.figure(figsize=(16,8))
colors = ['red', 'green', 'blue', 'black']
for i in range(numpy_data[0].size):
    ax = figure.add_subplot(4,4,i+1)
    ax.set_ylabel('Flower type')
    ax.set_xlabel('feature' + str(i+1))
    plt.scatter(numpy_data[:,i], numpy_labels, color=colors[i])
plt.show()