import tensorflow as tf
import torch
import numpy as np
from .poly import Poly

class Legendre(Poly):
	"""
	Legendre Polynomials
	"""
	def __init__(self, module, degree, x, dtype = 'float32', loglevel = 0, index_comb = None):
		"""
		input:
			module: 'tensorflow', 'pytorch', or 'numpy'
			degree: highest degree of polynomial
			x: a tensor of shape [Nsample*Nparameter], each row is a sample point, each column represents a parameter
			dtype: 'float32' or 'float64'
			loglevel: 1 to print time info and 0 to mute
			index_comb: combination of tensor product indices. If index_comb == None, the class will generate a new combination.
		"""

		if module == 'tensorflow':
			initial = [lambda x: tf.ones_like(x), lambda x: x]
			recurrence = lambda p1, p2, n, x: ((2*n+1)*tf.multiply(x, p1)-n*p2)/(n+1)
		elif module == 'pytorch':
			initial = [lambda x: torch.ones_like(x), lambda x: x]
			recurrence = lambda p1, p2, n, x: ((2*n+1)*x*p1-n*p2)/(n+1)
		elif module == 'numpy':
			initial = [lambda x: np.ones_like(x), lambda x: x]
			recurrence = lambda p1, p2, n, x: ((2*n+1)*x*p1-n*p2)/(n+1)
		Poly.__init__(self, module, degree, x, initial, recurrence, dtype, loglevel, index_comb)


class Legendre_Normalized(Poly):
	"""
	Normalized Legendre Polynomials with integral be 1 if n = m.
	"""
	def __init__(self, module, degree, x, dtype = 'float32', loglevel = 0, index_comb = None):
		"""
		input:
			module: 'tensorflow' or 'pytorch'
			degree: highest degree of polynomial
			x: a tensor of shape [Nsample*Nparameter], each row is a sample point, each column represents a parameter
			dtype: 'float32' or 'float64'
			loglevel: 1 to print time info and 0 to mute
			index_comb: combination of tensor product indices. If index_comb == None, the class will generate a new combination.
		"""

		if module == 'tensorflow':
			initial = [lambda x: tf.ones_like(x)*np.sqrt(1/2), lambda x: x*np.sqrt(3/2)]
			recurrence = lambda p1, p2, n, x: (np.sqrt((2*n+1)*(2*n+3))*tf.multiply(x, p1)-n*np.sqrt((2*n+3)/(2*n-1))*p2)/(n+1)
		elif module == 'pytorch':
			initial = [lambda x: torch.ones_like(x)*np.sqrt(1/2), lambda x: x*np.sqrt(3/2)]
			recurrence = lambda p1, p2, n, x: (np.sqrt((2*n+1)*(2*n+3))*x*p1-n*np.sqrt((2*n+3)/(2*n-1))*p2)/(n+1)
		elif module == 'numpy':
			initial = [lambda x: np.ones_like(x)*np.sqrt(1/2), lambda x: x*np.sqrt(3/2)]
			recurrence = lambda p1, p2, n, x: (np.sqrt((2*n+1)*(2*n+3))*x*p1-n*np.sqrt((2*n+3)/(2*n-1))*p2)/(n+1)
		Poly.__init__(self, module, degree, x, initial, recurrence, dtype, loglevel, index_comb)