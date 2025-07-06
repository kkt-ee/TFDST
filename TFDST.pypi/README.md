# TFDST: Fast Discrete Shearlet Transform layers in TensorFlow

<!-- [![PyPI Version](https://img.shields.io/pypi/v/TFDST?label=PyPI&color=gold)](https://pypi.org/project/TFDST/)  -->
<!-- [![PyPI Version](https://img.shields.io/pypi/pyversions/TFDST)](https://pypi.org/project/TFDST/) -->
[![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.15--2.19-darkorange)](https://www.tensorflow.org/)
[![Keras Version](https://img.shields.io/badge/keras-2--3-darkred)](https://keras.io/)
[![CUDA Version](https://img.shields.io/badge/cuda-12.5.1-green)](https://developer.nvidia.com/cuda-toolkit)
[![MIT](https://img.shields.io/badge/license-GPLv3-deepgreen.svg?style=flat)](https://github.com/kkt-ee/TFDST/LICENSE)


```python
from TFDST.DST2DFB import DST2D
from TFDST.DST3DFB import DST3D


## Example usage 2D
forward  = DST2D(N=N, J=2, L=[1, 2], B=[4, 8], norm=True, wave='bior1.5')
inverse = DST2D(N=N, J=2, L=[1, 2], B=[4, 8], norm=True, wave='bior1.5', transform='inverse')

## Example usage 3D
forward = DST3D(N=N, J=2, L=[1, 2], B=[4, 8], norm=True, wave='bior1.5')
inverse = DST3D(N=N, J=2, L=[1, 2], B=[4, 8], norm=True, wave='bior1.5', transform='inverse')
    
```

