from distutils.core import setup

setup(
	name = 'orthnet',
	version = '0.1',
	keywords = ['orthogonal polynomial', 'tensorflow', 'pytorch'],
	description = 'TensorFlow and PyTorch layers for generating orthogonal polynomials',
	license = 'MIT',
	author = 'Chuan Lu',
	author_email = 'chuan-lu@uiowa.edu',
	packages = ['orthnet', 'orthnet.tensorflow', 'orthnet.pytorch', 'orthnet.utils'],
	)