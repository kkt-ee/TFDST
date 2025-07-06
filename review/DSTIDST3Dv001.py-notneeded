# # include ../dirx 
mylibpath = [
    '/data1/kishoretarafdar/src.port/DST.v0/shearlet.layers'
    ]
import sys
[sys.path.insert(0,_) for _ in mylibpath]
del mylibpath
from ShearletTransform3Dv001 import ShearletTransform3D
import time

class DST3D(ShearletTransform3D):
    def __init__(self, transform=None, **kwargs):#, l1=0.0, l2=0.0):
        super(DST3D, self).__init__(**kwargs)
        self.transform = transform
        if self.transform == 'inverse':
            self.name = 'IDST3D'

    
    ## UPDATE applied for batched multichannel inputs
    def forward(self, x):
        x = tf.cast(x, dtype=tf.complex128)
        FB = self._yield_filter_bank_tensor()
        x = tf.transpose(x, perm=[0,4,1,2,3])
        xfft = tf.signal.fft3d(x)
        filtered = tf.einsum('bcijk,fijk->bcfijk', xfft, FB)
        filtered = tf.signal.ifft3d(filtered)
        if self.norm:
            self.norm_factors = tf.cast(self.norm_factors, dtype=tf.complex128)
            filtered *= self.norm_factors
            # filtered = tf.einsum('bcfijk,fijk->bcfijk', filtered, self.norm_factors)
        return filtered
    
    def call(self, x):
        if self.transform==None:
            return self.forward(x)
        elif self.transform=='inverse':
            return self.inverse(x)
        else:
            raise ValueError(f"Unknown key {transform}!! keys 'None' (default) or 'inverse' only allowed")

    def inverse(self, y):
        y = tf.cast(y, tf.complex128)
        self.norm_factors = tf.cast(self.norm_factors, tf.complex128)
        if self.norm:
            y = y/self.norm_factors
            # y = tf.einsum('bcfijk,fijk->bcfijk', y, 1/self.norm_factors)      
        yfft = tf.signal.fft3d(y)
        if 'bio' in self.wave:
            FB = self._yield_filter_bank_tensor_rec()
        else:
            FB = self._yield_filter_bank_tensor()
        synthesized = tf.einsum('bcfijk,fijk->bcijk', yfft, FB)
        synthesized = tf.cast(synthesized, tf.complex128)
        synthesized = tf.signal.ifft3d(synthesized)#, axes=(-3, -2, -1))
        synthesized = tf.transpose(synthesized, perm=[0,2,3,4,1])
        return tf.math.real(synthesized)


start_time = time.time()
# ST3D = ShearletTransform3D(N=128, J=2, L=[1, 2], B=[4, 8], norm=True)
ST3D = ShearletTransform3D(N=32, J=2, L=[1, 2], B=[4, 8], norm=True, wave='db6')
ST3D = ShearletTransform3D(N=32, J=2, L=[1, 2], B=[4, 8], norm=True, wave='db10')
# ST3D = DST3D(N=128, J=2, L=[1, 2], B=[4, 8], norm=True, wave='bior1.5')

# ST3D = ShearletTransform3D(N=32, J=3, L=[1, 4, 8], B=[6, 8, 10], norm=True, wave='db20')
print(time.time()-start_time)
     