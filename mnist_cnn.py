import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tensorflow.examples.tutorials.mnist import input_data
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2D(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


if __name__ == '__main__':

    lr = 1e-5
    epochs = 20000

    x = tf.placeholder(tf.float32, [None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    _y = tf.placeholder(tf.float32, [None, 10])
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2D(x_image, W_conv1)+b_conv1)
    h_pool1 = max_pooling_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2D(h_pool1, W_conv2)+b_conv2)
    h_pool2 = max_pooling_2x2(h_conv2)

    h_pool2_flatten = tf.reshape(h_pool2, [-1, 7*7*64])

    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flatten, W_fc1)+b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, 0.8)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=_y))
    optimizer = tf.train.AdamOptimizer(lr)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    for i in range(epochs):
        batch_x, batch_y = mnist.train.next_batch(100)
        if i%100 == 0:
            print(i, sess.run(accuracy, feed_dict={x: batch_x, _y: batch_y}))
        sess.run(train, feed_dict={x: batch_x, _y: batch_y})

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, _y: mnist.test.labels}))
