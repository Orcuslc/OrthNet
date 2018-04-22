# OrthNet
TensorFlow, PyTorch and Numpy layers for generating multi-dimensional Orthogonal Polynomials


[1. Installation](#installation)  
[2. Usage](#usage)  
[3. Polynomials](#polynomials)  
[4. Base Class(Poly)](#base-class)  


## Installation:
1. the stable version:  
`pip3 install orthnet`

2. the dev version:
```
git clone https://github.com/orcuslc/orthnet.git && cd orthnet
python3 setup.py build_ext --inplace && python3 setup.py install
```

## Usage:
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
L = Legendre(x, 5)
print(L.tensor)
```

### Specify Backend 
In some scenarios, users can specify the exact backend compatible with the input `x`. The backends provided are:
- [`orthnet.TensorflowBackend()`](./orthnet/backend/_tensorflow.py)
- [`orthnet.TorchBackend()`](./orthnet/backend/_torch.py)
- [`orthnet.NumpyBackend()`](./orthnet/backend/_numpy.py)

An example to specify the backend is as follows.
```python
import numpy as np
from orthnet import Legendre, NumpyBackend

x = np.random.random((10, 2))
L = Legendre(x, 5, backend = NumpyBackend())
print(L.tensor)
```

### Specify tensor product combinations
In some scenarios, users may provide pre-computed tensor product combinations to save computing time. An example of providing combinations is as follows.
```python
import numpy as np
from orthnet import Legendre, enum_dim

dim = 2
degree = 5
x = np.random.random((10, dim))
L = Legendre(x, degree, combinations = enum_dim(degree, dim))
print(L.tensor)
```

## Polynomials:  
| Class | Polynomial |  
|-------|-----------|
| [`orthnet.Legendre(Poly)`](./orthnet/poly/_legendre.py) | [Legendre polynomial](https://en.wikipedia.org/wiki/Legendre_polynomials) |  
| [`orthnet.Legendre_Normalized(Poly)`](./orthnet/poly/_legendre.py) | [Normalized Legendre polynomial](https://en.wikipedia.org/w/index.php?title=Legendre_polynomials&section=6#Additional_properties_of_Legendre_polynomials)  |  
| [`orthnet.Laguerre(Poly)`](./orthnet/poly/_laguerre.py) | [Laguerre polynomial](https://en.wikipedia.org/wiki/Laguerre_polynomials)  |  
| [`orthnet.Hermite(Poly)`](./orthnet/poly/_hermite.py) | [Hermite polynomial of the first kind (in probability theory)](https://en.wikipedia.org/wiki/Hermite_polynomials)  |  
| [`orthnet.Hermite2(Poly)`](./orthnet/poly/_hermite.py) | [Hermite polynomial of the second kind (in physics)](https://en.wikipedia.org/wiki/Hermite_polynomials)  |  
| [`orthnet.Chebyshev(Poly)`](./orthnet/poly/_chebyshev.py) | [Chebyshev polynomial of the first kind](https://en.wikipedia.org/wiki/Chebyshev_polynomials)  |  
| [`orthnet.Chebyshev2(Poly)`](./orthnet/poly/_chebyshev.py) | [Chebyshev polynomial of the second kind](https://en.wikipedia.org/wiki/Chebyshev_polynomials)  |  
| [`orthnet.Jacobi(Poly, alpha, beta)`](./orthnet/poly/_jacobi.py) | [Jacobi polynomial](https://en.wikipedia.org/wiki/Jacobi_polynomials) | 


## Base class:
Class [`Poly(x, degree, combination = None)`](./orthnet/poly/polynomial.py):
- Inputs:
    + `x` a tensor
    + `degree` highest degree for target polynomials
    + `combination` optional, tensor product combinations
- Attributes:
    + `Poly.tensor` the tensor of function values (with degree from 0 to `Poly.degree`(included))
    + `Poly.length` the number of function basis (columns) in `Poly.tensor`
    + `Poly.index` the index of the first combination of each degree in `Poly.combinations`
    + `Poly.combinations` all combinations of tensor product
    + `Poly.tensor_of_degree(degree)` return all polynomials of given degrees
    + `Poly.eval(coefficients)` return the function values with given coefficients
    + `Poly.quadrature(function, weight)` return Gauss quadrature with given function and weight
