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
L = Legendre(x, 2)
print(L.tensor)
```


## Polynomials:  
| Class | Polynomial |  
|-------|-----------|
| `orthnet.Legendre` | [Legendre polynomial](https://en.wikipedia.org/wiki/Legendre_polynomials) |  
| `orthnet.Legendre_Normalized(Poly)` | [Normalized Legendre polynomial](https://en.wikipedia.org/w/index.php?title=Legendre_polynomials&section=6#Additional_properties_of_Legendre_polynomials)  |  
| `orthnet.Laguerre(Poly)` | [Laguerre polynomial](https://en.wikipedia.org/wiki/Laguerre_polynomials)  |  
| `orthnet.Hermite(Poly)` | [Hermite polynomial of the first kind (in probability theory)](https://en.wikipedia.org/wiki/Hermite_polynomials)  |  
| `orthnet.Hermite2(Poly)` | [Hermite polynomial of the second kind (in physics)](https://en.wikipedia.org/wiki/Hermite_polynomials)  |  
| `orthnet.Chebyshev(Poly)` | [Chebyshev polynomial of the first kind](https://en.wikipedia.org/wiki/Chebyshev_polynomials)  |  
| `orthnet.Chebyshev2(Poly)` | [Chebyshev polynomial of the second kind](https://en.wikipedia.org/wiki/Chebyshev_polynomials)  |  
| `orthnet.Jacobi(Poly, alpha, beta)` | [Jacobi polynomial](https://en.wikipedia.org/wiki/Jacobi_polynomials) | 


## Base class:
Class `Poly(x, degree, combination = None)`:
- Inputs:
    + `x` a tensor
    + `degree` highest degree for target polynomials
    + `combination` optional, (if the combinations of some degree and dim is computed by `orthnet.enum_dim(degree, dim)`, then one may pass the combinations to save computing time).
- Attributes:
    + `Poly.tensor` the tensor of function values (with degree from 0 to `Poly.degree`(included))
    + `Poly.length` the number of function basis (columns) in `Poly.tensor`
    + `Poly.index` the index of the first combination of each degree in `Poly.combinations`
    + `Poly.combinations` all combinations of tensor product
    + `Poly.tensor_of_degree(degree)` return all polynomials of given degrees
    + `Poly.eval(coefficients)` return the function values with given coefficients
    + `Poly.quadrature(function, weight)` return Gauss quadrature with given function and weight
