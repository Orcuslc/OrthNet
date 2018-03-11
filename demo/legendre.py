import sys
sys.path.append('../')
from orthnet import Legendre
import tensorflow as tf
import torch
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# x0 = np.linspace(-1, 1, 1)
# x_data = np.zeros((1, 2))
# x_data[:, 0] = x0
# x_data[:, 1] = x0
x_data = np.array([[1., -1.], [1., -1.]])
x_tf = tf.placeholder(dtype = tf.float32, shape = (None, 2))
x_torch = Variable(torch.Tensor(x_data))

y_tf = Legendre('tensorflow', 2, x_tf)
y_torch = Legendre('pytorch', 2, x_torch)

with tf.Session() as sess:
	z_tf = sess.run(y_tf.tensor, feed_dict = {x_tf:x_data})
	# print(sess.run(y_tf.list, feed_dict = {x_tf:x_data}))

z_torch = y_torch.tensor.data.numpy()

print(z_tf)
print(z_torch)

y_tf.update(4)
y_torch.update(4)

with tf.Session() as sess:
	z_tf = sess.run(y_tf.tensor, feed_dict = {x_tf:x_data})
z_torch = y_torch.tensor.data.numpy()

print(z_tf)
print(z_torch)