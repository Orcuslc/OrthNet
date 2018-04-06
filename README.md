# OrthNet
TensorFlow, PyTorch and Numpy layers for generating multi-dimensional Orthogonal Polynomials

## Installation:
1. the stable version:  
`pip3 install orthnet`

2. the dev version:
```
git clone https://github.com/orcuslc/orthnet.git && cd orthnet
python3 setup.py build_ext --inplace && python3 setup.py install
```

---
## Orthogonal polynomials supported:
- [Legendre polynomial](https://en.wikipedia.org/wiki/Legendre_polynomials) with Normalized Legendre polynomial
- [Laguerre polynomial](https://en.wikipedia.org/wiki/Laguerre_polynomials)
- [Hermite polynomial](https://en.wikipedia.org/wiki/Hermite_polynomials)
  - hermite: He(x) in Wiki, the poly in probability theory
  - hermite2: H(x) in Wiki, the poly in physics
- [Chebyshev polynomial](https://en.wikipedia.org/wiki/Chebyshev_polynomials)
  - chebyshev: T(x) in Wiki, of the first kind
  - chebyshev2: U(x) in Wiki, of the second kind
- [Jacobi polynomial](https://en.wikipedia.org/wiki/Jacobi_polynomials)

---

## Classes:
- orthnet.Legendre(Poly)
- orthnet.Legendre_Normalized(Poly)
- orthnet.Laguerre(Poly)
- orthnet.Hermite(Poly)
- orthnet.Hermite2(Poly)
- orthnet.Chebyshev(Poly)
- orthnet.Chebyshev2(Poly)
- orthnet.Jacobi(Poly)

## Base class:
Class `Poly(module, degree, x, dtype = 'float32', loglevel = 0)`:

- Inputs:
    + module: one of {'tensorflow', 'pytorch', 'numpy'}
    + degree: the highest degree of target polynomial
    + x: input tensor of type {tf.placaholder, tf.Variable, torch.Variable, torch.Tensor, numpy.ndarray, numpy.matrix}
    + dtype: 'float32' or 'float64'
    + loglevel: 1 to print time cost and 0 to mute

- `Poly.tensor`: return a tensor of function values
- `Poly.combination`: return the combination of dimensions, in lexicographical order
- `Poly.index`: return the index of the first combination of each degree in `self.combination`
- `Poly.update(degree)`: update the degree of polynomial
- `Poly.get_combination(start, end):`: return the combination of degrees from `start`(included) till `end`(included)
- `Poly.get_poly(start, end)`: return the polynomials of degrees from `start`(included) till `end`(included)
- `Poly.eval(coefficients)`: evaluate the value of polynomial with coefficients
- `Poly.quadrature(func, weight)`: evaluate Gauss quadrature with target function and weights
