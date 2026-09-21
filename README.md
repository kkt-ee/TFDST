# TFDST: Fast Discrete Shearlet Transform Layers in TensorFlow

[![PyPI Version](https://img.shields.io/pypi/v/TFDST?label=PyPI&color=gold)](https://pypi.org/project/TFDST/)
[![Python Versions](https://img.shields.io/pypi/pyversions/TFDST)](https://pypi.org/project/TFDST/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-required-darkorange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-deepgreen.svg?style=flat)](https://github.com/kkt-ee/TFDST/LICENSE)

`TFDST` provides differentiable TensorFlow/Keras layers for fast 2D and 3D
discrete shearlet transforms and their inverse transforms.



## Capabilities

- `DST2D` / `IDST2D` through `transform=None` or `transform="inverse"`
- `DST3D` / `IDST3D` through `transform=None` or `transform="inverse"`
- Batched multichannel TensorFlow tensors
- Backpropagation through the transform layers

## Limitations

- Inputs are expected to match the configured spatial size `N`.
- 2D input shape: `[batch, N, N, channels]`.
- 3D input shape: `[batch, N, N, N, channels]`.
- Wavelet filter coefficients are provided through `PyWavelets`.

## Installation

```bash
pip install TFDST
```

## Minimal Example

```python
import tensorflow as tf
from TFDST.DST2DFB import DST2D
from TFDST.DST3DFB import DST3D

N = 32

# 2D
x2 = tf.random.normal([1, N, N, 1])
dst2 = DST2D(N=N, J=2, L=[1, 2], B=[4, 8], norm=True, wave="bior1.5")
idst2 = DST2D(N=N, J=2, L=[1, 2], B=[4, 8], norm=True, wave="bior1.5", transform="inverse")
y2 = dst2(x2)
r2 = idst2(y2)

# 3D
x3 = tf.random.normal([1, N, N, N, 1])
dst3 = DST3D(N=N, J=2, L=[1, 2], B=[4, 8], norm=True, wave="bior1.5")
idst3 = DST3D(N=N, J=2, L=[1, 2], B=[4, 8], norm=True, wave="bior1.5", transform="inverse")
y3 = dst3(x3)
r3 = idst3(y3)

print(x2.shape, r2.shape)
print(x3.shape, r3.shape)
```

## Arguments

| Argument | Meaning |
|---|---|
| `N` | Spatial size of each axis; 2D uses `N x N`, 3D uses `N x N x N`. |
| `J` | Number of multiscale shearlet levels. |
| `L` | Shear parameters per level; must have length `J`. |
| `B` | Bandwidth/window parameters per level; must have length `J`. |
| `wave` | PyWavelets wavelet name used for the radial wavelet filters, e.g. `"bior1.5"`. |
| `norm` | If `True`, applies filter-bank normalization. |
| `transform` | Use `None` for forward DST and `"inverse"` for inverse DST. |

## Citation

This software is released for broad research, educational, and engineering use. If this package helps your work, please cite the following paper:

```bibtex
@misc{tarafdar2026interpretablefrugallearningsystems,
      title={Interpretable and Frugal Learning Systems Employing Multiresolution Pyramids and Volterra Kernels},
      author={Kishore Kumar Tarafdar},
      year={2026},
      eprint={2606.15011},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2606.15011},
}
```



## License

Apache License 2.0. See [`LICENSE`](LICENSE).


## Credit 
1. **Vineet Ghule**: Basic shearlet filter-bank construction with perfect reconstruction. 
2. **Kishore Kumar Tarafdar:** Realization as TensorFlow/Keras layers with batched multichannel I/O with biothogonal wavelet support and performance-oriented updates.

* * *

***TFDST (C) 2025 Vineet Ghule and Kishore Kumar Tarafdar, भारत*** 🇮🇳