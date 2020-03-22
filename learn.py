import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# keras is API for tensorflow

# data loading starts here

data = keras.datasets.fashion_mnist

# divide data into training and test data
# 80 - 90 % data for training, the rest will be for testing data

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# this data has 10 labels

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# the value of output will be
# 0 for T-shirt/top
# 1 for Trouser
# 2 for Pullover
# and so on, hence we've got 10 nodes on the output

train_images = train_images / 255.0
test_images = test_images / 255.0


print(train_images[7])

plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()

# data loading finished here

# our input data is 28 x 28 pixel (2 dimensional array)

# now we want to make it into 1 dimensional array and becomes 1 x 784, where 784 is 28 x 28

# thus, we will have 784 neurons as input


