import tensorflow as tf
import torch
import numpy as np
from .poly import Poly

class Hermite(Poly):
	"""
	Hermite Polynomials of the first kind (in probability theory)
	"""
	def __init__(self, module, degree, x, *args, **kw):
		"""
		input:
			module: ['tensorflow', 'torch', 'numpy']
			degree: highest degree of polynomial
			x: a tensor of shape [Nsample*Nparameter], each row is a sample point, each column represents a parameter
		"""

		if module == 'tensorflow':
			initial = [lambda x: tf.ones_like(x), lambda x: x]
			recurrence = lambda p1, p2, n, x: tf.multiply(x, p1) - n*p2
		elif module == 'torch':
			initial = [lambda x: torch.ones_like(x), lambda x: x]
			recurrence = lambda p1, p2, n, x: x*p1 - p2*n
		elif module == 'numpy':
			initial = [lambda x: np.ones_like(x), lambda x: x]
			recurrence = lambda p1, p2, n, x: x*p1 - n*p2			
		Poly.__init__(self, module, degree, x, initial, recurrence, *args, **kw)
		

class Hermite2(Poly):
	"""
	Hermite Polynomials of the second kind (in Physics)
	"""
	def __init__(self, module, degree, x, *args, **kw):
		"""
		input:
			module: ['tensorflow', 'torch', 'numpy']
			degree: highest degree of polynomial
			x: a tensor of shape [Nsample*Nparameter], each row is a sample point, each column represents a parameter
		"""

		if module == 'tensorflow':
			initial = [lambda x: tf.ones_like(x), lambda x: 2*x]
			recurrence = lambda p1, p2, n, x: 2*(tf.multiply(x, p1) - n*p2)
		elif module == 'torch':
			initial = [lambda x: torch.ones_like(x), lambda x: x*2]
			recurrence = lambda p1, p2, n, x: (x*p1 - p2*n)*2
		elif module == 'numpy':
			initial = [lambda x: np.ones_like(x), lambda x: 2*x]
			recurrence = lambda p1, p2, n, x: 2*(x*p1 - n*p2)
		Poly.__init__(self, module, degree, x, initial, recurrence, *args, **kw)
