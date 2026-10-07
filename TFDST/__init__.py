"""TFDST: Fast Discrete Shearlet Transform Layers in TensorFlow.
Copyright 2025 Kishore Kumar Tarafdar.
Licensed under the Apache License, Version 2.0. See LICENSE for details.

Filter-bank construction follows mathematical structure developed with
credit to Vineet Ghule. This implementation provides differentiable
TensorFlow/Keras layers with batched multichannel support and
performance-oriented updates.
"""




# import tensorflow as tf

# from TFDST.DST2DFB import DST2D
# from TFDST.DST3DFB import DST3D

## Example usage 2D
# forward  = DST2D(N=N, J=2, L=[1, 2], B=[4, 8], norm=True, wave='bior1.5')
# inverse = DST2D(N=N, J=2, L=[1, 2], B=[4, 8], norm=True, wave='bior1.5', transform='inverse')

## Example usage 3D
# forward = DST3D(N=N, J=2, L=[1, 2], B=[4, 8], norm=True, wave='bior1.5')
# inverse = DST3D(N=N, J=2, L=[1, 2], B=[4, 8], norm=True, wave='bior1.5', transform='inverse')

__version__="0.0.2"
