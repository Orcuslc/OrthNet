import tensorflow as tf
import torch
from ..utils.poly import Poly1d, Poly

class Legendre(Poly):
	"""
	Legendre Polynomials
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
			recurrence = lambda p1, p2, n, x: ((2*n+1)*tf.multiply(x, p1)-n*p2)/(n+1)
		elif module == 'pytorch':
			initial = [lambda x: torch.ones_like(x), lambda x: x]
			recurrence = lambda p1, p2, n, x: ((2*n+1)*x*p1-n*p2)/(n+1)
		