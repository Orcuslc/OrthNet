import tensorflow as tf
import torch
import numpy as np
from .poly import Poly

class Laguerre(Poly):
	"""
	Laguerre Polynomials
	"""
	def __init__(self, module, degree, x, *args, **kw):
		"""
		input:
			module: ['tensorflow', 'torch', 'numpy']
			degree: highest degree of polynomial
			x: a tensor of shape [Nsample*Nparameter], each row is a sample point, each column represents a parameter
		"""
		if module == 'tensorflow':
			initial = [lambda x: tf.ones_like(x), lambda x: 1-x]
			recurrence = lambda p1, p2, n, x: (tf.multiply(2*n+1-x, p1)-n*p2)/(n+1)
		elif module == 'torch':
			initial = [lambda x: torch.ones_like(x), lambda x: 1-x]
			recurrence = lambda p1, p2, n, x: (p1*(2*n+1-x)-p2*n)/(n+1)
		elif module == 'numpy':
			initial = [lambda x: np.ones_like(x), lambda x: 1-x]
			recurrence = lambda p1, p2, n, x: ((2*n+1-x)* p1-n*p2)/(n+1)
		Poly.__init__(self, module, degree, x, initial, recurrence, *args, **kw)
