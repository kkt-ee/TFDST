import tensorflow as tf
from TFDST.ShearletTransform2Dlayout import ShearletTransform2D

class DST2D(ShearletTransform2D):
    """Fast DST 2D and IDST 2D layer
    
    TFDST: Fast Discrete Shearlet Transform Layers in TensorFlow.
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
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    """
    def __init__(self, transform=None, **kwargs):#, l1=0.0, l2=0.0):
        super(DST2D, self).__init__(**kwargs)
        self.transform = transform
        if self.transform == 'inverse':
            self.name = 'IDST2D'
    
    ## UPDATE applied for batched multichannel inputs
    def forward(self, x):
        x = tf.cast(x, dtype=tf.complex128)
        # FB = self._yield_filter_bank_tensor()
        FB, _ = self.getFB()
        x = tf.transpose(x, perm=[0,3,1,2])
        xfft = tf.signal.fft2d(x)
        filtered = tf.einsum('bcij,fij->bcfij', xfft, FB)
        filtered = tf.signal.ifft2d(filtered)
        if self.norm:
            # self.norm_factors = tf.cast(self.norm_factors, dtype=tf.complex128)
            filtered *= self.norm_factors
            # filtered = tf.einsum('bcfijk,fijk->bcfijk', filtered, self.norm_factors)
        y = tf.transpose(filtered, perm=[0, 3,4, 1,2])
        ## reshape last two axis and output
        shape_tmp = tf.shape(y)
        y = tf.reshape(y, (shape_tmp[0], shape_tmp[1], shape_tmp[2], shape_tmp[3]*shape_tmp[4]))
        return y
    
    def call(self, x):
        if self.transform==None:
            return self.forward(x)
        elif self.transform=='inverse':
            return self.inverse(x)
        else:
            raise ValueError(f"Unknown key {transform}!! keys 'None' (default) or 'inverse' only allowed")

    def inverse(self, y):
        y = tf.cast(y, tf.complex128)
        ## reshape to (batch, n1,n2, channels, shearlets)
        input_shape = tf.shape(y)
        num_shearlet_filters = self.norm_factors.shape[0]
        channels = input_shape[-1] // num_shearlet_filters
        y = tf.reshape(y, (input_shape[0],input_shape[1],input_shape[2], channels, num_shearlet_filters))
        ## transpose to (batch, channels, shearlets, n1,n2,n3) 
        ## since fft3d works on last three axis
        y = tf.transpose(y, perm=[0, 3,4, 1,2])
        # self.norm_factors = tf.cast(self.norm_factors, tf.complex128)
        if self.norm:
            y = y/self.norm_factors
            # y = tf.einsum('bcfijk,fijk->bcfijk', y, 1/self.norm_factors)      
        yfft = tf.signal.fft2d(y)
        if 'bio' in self.wave:
            _, FB = self.getFB()
        else:
            FB, _ = self.getFB()  
        synthesized = tf.einsum('bcfij,fij->bcij', yfft, FB)
        synthesized = tf.cast(synthesized, tf.complex128)
        synthesized = tf.signal.ifft2d(synthesized)#, axes=(-3, -2, -1))
        synthesized = tf.transpose(synthesized, perm=[0,2,3,1])
        return tf.math.real(synthesized)


if __name__=='__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
    # start_time = time.time()
    # ST3D = ShearletTransform3D(N=128, J=2, L=[1, 2], B=[4, 8], norm=True)
    # ST3D = DST3D(N=32, J=2, L=[1, 2], B=[4, 8], norm=True, wave='db6')
    # ST3D = DST3D(N=32, J=2, L=[1, 2], B=[4, 8], norm=True, wave='db10')
    # ST3D = DST3D(N=128, J=2, L=[1, 2], B=[4, 8], norm=True, wave='bior1.5')

    # ST3D = ShearletTransform3D(N=32, J=3, L=[1, 4, 8], B=[6, 8, 10], norm=True, wave='db20')
    # print(time.time()-start_time)
    
    ## Example 1: model summary
    import numpy as np
    n = 64
    axis1 = np.arange(0,n)
    x = np.einsum('i,j->ij', axis1, axis1)
    # x = np.einsum('i,j,k->ijk', axis1, axis1, axis1)
    xx = tf.expand_dims(tf.expand_dims(x, axis=-1), axis=0)
    xx = tf.concat([xx,xx, xx], axis=-1)
    xx.shape
    # viz(x)
    # dst3D = DST3D(N=n, J=2, L=[1, 2], B=[4, 8], norm=True, wave='db10')
    dst2D = DST2D(N=n, J=2, L=[1, 2], B=[4, 8], norm=True, wave='bior1.5')
    # dst3D = DST3D(N=n, J=2, L=[1, 2], B=[4, 8], norm=True, wave='rbio1.5')
    dst2D.forward(xx).shape
    xxrec = dst2D.inverse(dst2D(xx))
    xxrec.dtype, xxrec.shape
    print(f"\nReconstruction error: {tf.reduce_sum(tf.abs(xxrec - tf.cast(xx,dtype=tf.float64)))}\n")

    ## Example 2: Sample functional model
    N, channels = 16, 2
    input_shape = (N, N, channels)  # Replace N with the actual size of x    #3D
    inputs = tf.keras.Input(shape=input_shape)

    H  = DST2D(N=N, J=2, L=[1, 2], B=[4, 8], norm=True, wave='bior1.5')
    Hr = DST2D(N=N, J=2, L=[1, 2], B=[4, 8], norm=True, wave='bior1.5', transform='inverse')
    q = H(inputs)
    outputs = Hr(q)
    # outputs = q
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', jit_compile=False)
    model.summary()

    ## 3D Random data
    inputs_data = tf.random.normal((1, N, N, channels))
    targets = tf.random.normal((1, N, N, channels))
    # Training loop for 5 epochs
    epochs=5
    # for epoch in range(5):
    history = model.fit(inputs_data, targets, epochs=5, verbose=1)