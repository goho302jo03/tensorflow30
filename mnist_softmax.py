import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from tensorflow.examples.tutorials.mnist import input_data
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

lr = 0.5
epoch = 5000

train_img   = mnist.train.images
train_label = mnist.train.labels
vali_img    = mnist.validation.images
vali_label  = mnist.validation.labels
test_images = mnist.test.images
test_label = mnist.test.labels

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
_y = tf.placeholder(tf.float32, [None, 10])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=_y))
optimizer = tf.train.GradientDescentOptimizer(lr)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(epoch):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x: batch_x, _y: batch_y})

correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, _y: mnist.test.labels}))


