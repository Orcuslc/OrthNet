import tensorflow as tf
import torch
from ..utils import Poly1d, Poly

class Hermite(Poly):
	"""
	Hermite Polynomials of the first kind (in probability theory)
	"""
	def __init__(self, module, degree, x):
		"""
		input:
			module: 'tensorflow' or 'pytorch'
			degree: highest degree of polynomial
			x: a tensor of shape [Nsample*Nparameter], each row is a sample point, each column represents a parameter
		"""

		if module == 'tensorflow':
			initial = [lambda x: tf.ones_like(x), lambda x: x]
			recurrence = lambda p1, p2, n, x: tf.multiply(x, p1) - n*p2
		elif module == 'pytorch':
			initial = [lambda x: torch.ones_like(x), lambda x: x]
			recurrence = lambda p1, p2, n, x: x*p1 - n*p2
		Poly.__init__(self, module, degree, x, initial, recurrence)
		

class Hermite2(Poly):
	"""
	Hermite Polynomials of the second kind (in Physics)
	"""
	def __init__(self, module, degree, x):
		"""
		input:
			module: 'tensorflow' or 'pytorch'
			degree: highest degree of polynomial
			x: a tensor of shape [Nsample*Nparameter], each row is a sample point, each column represents a parameter
		"""

		if module == 'tensorflow':
			initial = [lambda x: tf.ones_like(x), lambda x: 2*x]
			recurrence = lambda p1, p2, n, x: 2*(tf.multiply(x, p1) - n*p2)
		elif module == 'pytorch':
			initial = [lambda x: torch.ones_like(x), lambda x: 2*x]
			recurrence = lambda p1, p2, n, x: 2*(x*p1 - n*p2)
		Poly.__init__(self, module, degree, x, initial, recurrence)