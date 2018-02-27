import sys
sys.path.append('../../')
from OrthNet.pytorch import chebyshev_tensor, multi_dim_chebyshev_tensor
import torch
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

order1 = 5
order2 = 3

x1_data = np.linspace(-1, 1, 100).reshape((-1, 1))
x2_data = np.linspace(-1, 1, 100).reshape((-1, 1))

x1 = Variable(torch.Tensor(x1_data))
x2 = Variable(torch.Tensor(x2_data))

y1 = chebyshev_tensor(n = order1, x = x1)
y2 = multi_dim_chebyshev_tensor(n = order2, var = [x1, x2])

z1 = y1.data.numpy()
z2 = y2.data.numpy()

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