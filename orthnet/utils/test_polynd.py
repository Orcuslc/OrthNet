from poly import Poly1d, Poly
import tensorflow as tf
import torch
from torch.autograd import Variable
import numpy as np

tf_initial = [lambda x: tf.ones_like(x), lambda x: x]
tf_recurrence = lambda p1, p2, n, x: ((2*n+1)*tf.multiply(x, p1)-n*p2)/(n+1)
tf_x = tf.placeholder(dtype = tf.float32, shape=[None, 2])
tf_legendre = Poly('tensorflow', 1, tf_x, tf_initial, tf_recurrence)
tf_x_data = np.linspace(-1, 1, 4).reshape((-1, 2))

with tf.Session() as sess:
	y = sess.run(tf_legendre.tensor, feed_dict = {tf_x: tf_x_data})
print(tf_legendre.length)

tf_legendre.update(4)
with tf.Session() as sess:
	print(sess.run(tf_legendre.tensor, feed_dict = {tf_x: tf_x_data}))
print(tf_legendre.length)

# import time
# t1 = time.time()
# x = tf_legendre.combination
# t2 = time.time()
# x = tf_legendre.combination
# t3 = time.time()
# print(x)

# print('t1:', t2-t1)
# print('t2:', t3-t2)