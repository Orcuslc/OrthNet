import tensorflow as tf
import torch
import numpy as np
from .poly import Poly

class Chebyshev(Poly):
	"""
	Chebyshev Polynomials of the first kind
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
			recurrence = lambda p1, p2, n, x: 2*tf.multiply(x, p1) - p2
		elif module == 'torch':
			initial = [lambda x: torch.ones_like(x), lambda x: x]
			recurrence = lambda p1, p2, n, x: x*p1*2-p2
		elif module == 'numpy':
			initial = [lambda x: np.ones_like(x), lambda x: x]
			recurrence = lambda p1, p2, n, x: 2*x*p1-p2
		Poly.__init__(self, module, degree, x, initial, recurrence, *args, **kw)


class Chebyshev2(Poly):
	"""
	Chebyshev Polynomials of the second kind
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
			recurrence = lambda p1, p2, n, x: 2*tf.multiply(x, p1)-p2
		elif module == 'torch':
			initial = [lambda x: torch.ones_like(x), lambda x: x*2]
			recurrence = lambda p1, p2, n, x: x*p1*2-p2
		elif module == 'numpy':
			initial = [lambda x: np.ones_like(x), lambda x: 2*x]
			recurrence = lambda p1, p2, n, x: 2*x*p1-p2
		Poly.__init__(self, module, degree, x, initial, recurrence, *args, **kw)
