import tensorflow as tf
import numpy as np

#generate data
data_x = np.random.rand(100).astype(np.float32)
data_y = 0.1*data_x + 0.5

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
y = W*data_x + b

# Minimize the mean squared errors
loss = tf.reduce_mean(tf.square(y-data_y))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(601):
    sess.run(train)
    if step%50==0:
        print(step, sess.run(W), sess.run(b), sess.run(loss))
