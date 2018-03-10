import tensorflow as tf
import torch
from ..utils import Poly1d, Poly

class Chebyshev(Poly):
	"""
	Chebyshev Polynomials of the first kind
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
			recurrence = lambda p1, p2, n, x: 2*x*p1-p2
		Poly.__init__(self, module, degree, x, initial, recurrence)


class Chebyshev2(Poly):
	"""
	Chebyshev Polynomials of the second kind
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
			recurrence = lambda p1, p2, n, x: 2*tf.multiply(x, p1)-p2
		elif module == 'pytorch':
			initial = [lambda x: torch.ones_like(x), lambda x: 2*x]
			recurrence = lambda p1, p2, n, x: 2*x*p1-p2
		Poly.__init__(self, module, degree, x, initial, recurrence)