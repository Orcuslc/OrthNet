from distutils.core import setup, Extension

ext = Extension('orthnet.utils._enum_dim',
	sources = ['orthnet/utils/enum_dim/enum_dim.i'],
	language = 'c++',
	swig_opts = ['-c++'],
	)

setup(
	name = 'orthnet',
	version = '0.2.0',
	keywords = ['orthogonal polynomial', 'tensorflow', 'pytorch'],
	description = 'TensorFlow and PyTorch layers for generating orthogonal polynomials',
	license = 'MIT',
	author = 'Chuan Lu',
	author_email = 'chuan-lu@uiowa.edu',
	ext_modules = [ext],
	packages = ['orthnet', 'orthnet.poly', 'orthnet.utils'],
	)