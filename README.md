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
Class `Poly(x, degree, combination = None)`:
- Inputs:
    + `x`: a tensor
    + `degree`: highest degree for target polynomials
    + `combination`: optional, (if the combinations of some degree and dim is computed by `orthnet.enum_dim(degree, dim)`, then one may pass the combinations to save computing time).
- Attributes:
    + `Poly.tensor` the tensor of function values (with degree from 0 to `Poly.degree`(included))
    + `Poly.length` the number of function basis (columns) in `Poly.tensor`
    + `Poly.index` the index of the first combination of each degree in `Poly.combinations`
    + `Poly.combinations` all combinations of tensor product
    + `Poly.tensor_of_degree(degree)` all polynomials of some degrees
    + `Poly.eval(coefficients)` eval the function values with given coefficients
    + `Poly.quadrature(function, weight)` perform Gauss quadrature with given function and weight

## Examples:

### with TensorFlow
```python
import tensorflow as tf
import numpy as np
from orthnet import Legendre

x_data = np.random.random((10, 2))
x = tf.placeholder(dtype = tf.float32, shape = [None, 2])
L = Legendre(x, 5)

with tf.Session() as sess:
    print(L.tensor, feed_dict = {x: x_data})
```

### with PyTorch
```python
import torch
import numpy as np
from orthnet import Legendre

x = torch.DoubleTensor(np.random.random((10, 2)))
L = Legendre(x, 5)
print(L.tensor)
```

### with Numpy
```python
import numpy as np
from orthnet import Legendre

x = np.random.random((10, 2))
L = Legendre(x, 2)
print(L.tensor)
```

