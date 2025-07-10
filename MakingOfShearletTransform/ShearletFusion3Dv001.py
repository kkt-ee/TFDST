import numpy as np
import pywt
# from scipy.signal import freqz
# import matplotlib.pyplot as plt
import time
import math

import tensorflow as tf
# # include ../dirx 
# mylibpath = [
#     '/data1/kishoretarafdar/src.port/DST.v0/shearlet.layers/shearlet_transform.core'
#     ]
# import sys
# [sys.path.insert(0,_) for _ in mylibpath]
# del mylibpath

# import tensorflow as tf
from TFDST.ShearletTransform3Dlayout import ShearletTransform3D

class ShearletFusion3D(ShearletTransform3D):
    """TFDST: Fast Discrete Shearlet Transform Layers in TensorFlow.
    Copyright (C) 2025 Vineet Ghule and Kishore Kumar Tarafdar

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>."""
    
    def __init__(
        self, 
        transform=None, ## None or inverse
        **kwargs):#, l1=0.0, l2=0.0):
        super(ShearletFusion3D, self).__init__(**kwargs)
        
    def build(self, input_shape):
        in_channels = input_shape[-1]
        # self.N = input_shape[1]

    ## UPDATE 2!!
    def forward(self, x):
        x = tf.cast(x, dtype=tf.complex128)
        FB, _ = self.getFB()
        x = tf.transpose(x, perm=[0,4,1,2,3])
        xfft = tf.signal.fft3d(x)
        # FB = tf.transpose(FB, perm=[1,2,3,0])
        filtered = tf.einsum('bcijk,fijk->bcfijk', xfft, FB)
        filtered = tf.signal.ifft3d(filtered)
        
        if self.norm:
            print(self.norm_factors.shape, filtered.shape, '++')
            # self.norm_factors = tf.expand_dims(self.norm_factors, axis=-5)
            # self.norm_factors = tf.cast(self.norm_factors, dtype=tf.complex128)
            filtered *= self.norm_factors
        # filtered = tf.transpose(filtered, perm=[0,2,3,4,1])
        return filtered

    def inverse(self, y):
        y = tf.cast(y, tf.complex128)
        self.norm_factors = tf.cast(self.norm_factors, tf.complex128)
        if self.norm:
            y = y/self.norm_factors      
        yfft = tf.signal.fft3d(y)
        if 'bio' in self.wave:
            _, FB = self.getFB()#_yield_filter_bank_tensor_rec()
        else:
            FB, _ = self.getFB()#_yield_filter_bank_tensor()
        # synthesized = tf.einsum('bcfijk,fijk->bijk', yfft, FB)
        synthesized = tf.einsum('bcfijk,fijk->bfijk', yfft, FB)   ## trick time-space localization
        synthesized = tf.cast(synthesized, tf.complex128)
        synthesized = tf.signal.ifft3d(synthesized)#, axes=(-3, -2, -1))
        # synthesized = tf.expand_dims(synthesized, axis=-1)
        synthesized = tf.transpose(synthesized, perm=[0, 2,3,4, 1])
        return tf.math.real(synthesized)

    def call(self, x):
        ## fuse 
        q = self.forward(x)
        return self.inverse(q)

if __name__=='__main__':
    start_time = time.time()
    # ST3D = ShearletTransform3D(N=128, J=2, L=[1, 2], B=[4, 8], norm=True)
    # ST3D = ShearletTransform3D(N=32, J=2, L=[1, 2], B=[4, 8], norm=True, wave='db6')
    ST3D = ShearletFusion3D(N=128, J=2, L=[1, 2], B=[4, 8], norm=True, wave='bior1.5')

    # ST3D = ShearletTransform3D(N=32, J=3, L=[1, 4, 8], B=[6, 8, 10], norm=True, wave='db20')
    print(time.time()-start_time)
    del ST3D 

    # ## Example
    # import numpy as np
    # n = 64
    # axis1 = np.arange(0,n)
    # x = np.einsum('i,j->ij', axis1, axis1)
    # x = np.einsum('i,j,k->ijk', axis1, axis1, axis1)
    # xx = tf.expand_dims(tf.expand_dims(x, axis=-1), axis=0)
    # xx = tf.concat([xx,xx, xx], axis=-1)
    # xx.shape
    # # viz(x)
    # dst3D = ShearletFusion3D(N=n, J=2, L=[1, 2], B=[4, 8], norm=True, wave='rbio1.5')
    # dst3D.forward(xx).shape
    # xxrec = dst3D.inverse(dst3D.forward(xx))
    # xxrec.dtype, xxrec.shape
    # print(f"\nReconstruction error: {tf.reduce_sum(tf.abs(xxrec - tf.cast(xx,dtype=tf.float64)))}\n")

    # ## Example
    # # Define input shape and build the model for summary
    # # input_shape = (16,16, 2)  # Replace N with the actual size of x
    # n = 64
    # input_shape = (n, n, n, 12)
    # inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32)

    # # Create an instance of the custom layer
    # H = ShearletFusion3D(N=n, J=2, L=[1, 2], B=[4, 8], norm=True, wave='bior1.5')

    # # Apply the custom layer to the inputs
    # outputs = H(inputs)

    # # Build the model
    # model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # # Print the model summary
    # model.summary()

     ## Example 2: Sample functional model
    N, channels = 16, 256
    input_shape = (N, N, N, channels)  # Replace N with the actual size of x    #3D
    inputs = tf.keras.Input(shape=input_shape)

    # Create an instance of the custom layer
    H = ShearletFusion3D(N=N, J=2, L=[1, 2], B=[4, 8], norm=True, wave='bior1.5')
    outputs = H(inputs)
    # outputs = q
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', jit_compile=False)
    model.summary()

    ## 3D Random data
    inputs_data = tf.random.normal((1, N, N, N, channels))
    targets = tf.random.normal((1, N, N, N, 63))
    # Training loop for 5 epochs
    epochs=5
    # for epoch in range(5):
    history = model.fit(inputs_data, targets, epochs=5, verbose=1)