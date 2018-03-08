from poly import Poly1d
import tensorflow as tf
import torch
from torch.autograd import Variable
import numpy as np

tf_initial = [lambda x: tf.ones_like(x), lambda x: x]
tf_recurrence = lambda p1, p2, n, x: ((2*n+1)*tf.multiply(x, p1)-n*p2)/(n+1)
tf_x = tf.placeholder(dtype = tf.float32)
tf_legendre = Poly1d('tensorflow', 3, tf_x, tf_initial, tf_recurrence)
tf_x_data = np.linspace(-1, 1, 5).reshape((-1, 1))
with tf.Session() as sess:
	print(sess.run([tf_legendre.tensor], feed_dict = {tf_x: tf_x_data})[0])
tf_legendre.update(4)
with tf.Session() as sess:
	y = sess.run([tf_legendre.tensor], feed_dict = {tf_x: tf_x_data})[0]


# x1_data = np.linspace(-1, 1, 5).reshape((-1, 1))
# x1 = Variable(torch.Tensor(x1_data))
# initial = [lambda x: torch.ones_like(x), lambda x: x]
# recurrence = lambda p1, p2, n, x: ((2*n+1)*x*p1-n*p2)/(n+1)
# y1 = Poly1d('pytorch', 3, x1, initial, recurrence)
# print(y1.tensor)
# y1.update(4)
# print(y1.tensor)