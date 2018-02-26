import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_img   = mnist.train.images
train_label = mnist.train.labels
vali_img    = mnist.validation.images
vali_label  = mnist.validation.labels
test_images = mnist.test.images
test_label = mnist.test.labels


