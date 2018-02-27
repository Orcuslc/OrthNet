import sys
sys.path.append('../../')
from OrthNet.tensorflow import chebyshev_tensor, multi_dim_chebyshev_tensor
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

order1 = 5
order2 = 3

x1 = tf.placeholder(dtype = tf.float32)
x2 = tf.placeholder(dtype = tf.float32)

y1 = chebyshev_tensor(n = order1, x = x1)
y2 = multi_dim_chebyshev_tensor(n = order2, var = [x1, x2])

x1_data = np.linspace(-1, 1, 100).reshape((-1, 1))
x2_data = np.linspace(-1, 1, 100).reshape((-1, 1))

with tf.Session() as sess:
	z1 = sess.run([y1], feed_dict = {x1: x1_data})[0]
	z2 = sess.run([y2], feed_dict = {x1: x1_data, x2: x2_data})[0]

fig1 = plt.figure()
ax1 = fig1.gca()

for i in range(order1+1):
	ax1.plot(x1_data, z1[:, i], label = 'n = '+str(i))
ax1.legend()
ax1.grid(True)

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
x1_data, x2_data = np.meshgrid(x1_data, x2_data)
ax2.plot_surface(X = x1_data, Y = x2_data, Z = z2[:, -2])

plt.show()