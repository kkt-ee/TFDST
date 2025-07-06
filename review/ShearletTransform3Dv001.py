# import numpy as np
# from scipy.signal import freqz
# import matplotlib.pyplot as plt
# import math

import tensorflow as tf
import numpy as np # for π
import pywt
import time



class ShearletTransform3D(tf.keras.layers.Layer):
    def __init__(
        self, 
        N, 
        J, 
        L=None,  ## array of +ve integers of length J 
        B=None, 
        a=None, 
        norm=False, 
        wave='db3', 
        transform=None, ## inverse
        **kwargs):#, l1=0.0, l2=0.0):
        super(ShearletTransform3D, self, **kwargs).__init__()
        self.transform = transform
        self.π = tf.constant(np.pi, dtype=tf.float64)
        if a is not None:
            L = [int(tf.math.round(a**(2/3*j))) for j in range(J)]
            B = [a**(j+1)/L[j] for j in range(J)]
        self.N = N
        self.J = J
        self.L = L
        self.B = B
        self.wave = wave
        self.norm = norm
        self.shearlet_system = self.get_shearlet_system('wavedec')
        if 'bio' in self.wave:
            self.shearlet_system_rec = self.get_shearlet_system('waverec')
        if self.norm:
            self.norm_factors = [tf.reduce_sum(tf.math.square(tf.math.abs(filters)), axis=(-3, -2, -1), keepdims=True) for filters in self._yield_filters()]
            self.norm_factors = tf.concat(self.norm_factors, axis=-4)
            self.norm_factors = tf.math.sqrt(N**3/self.norm_factors)
            self.norm_factors = tf.cast(self.norm_factors, tf.complex128)
        
    @tf.function
    def v(self, x):
        x = tf.cast(x, tf.float64)  # Ensure consistent dtype if needed

        # Constants casted to match x's dtype
        ten = tf.constant(10.0, dtype=x.dtype)
        fifteen = tf.constant(15.0, dtype=x.dtype)
        six = tf.constant(6.0, dtype=x.dtype)
        one = tf.constant(1.0, dtype=x.dtype)
        zero = tf.constant(0.0, dtype=x.dtype)

        # Conditions
        cond0 = x <= zero
        cond1 = x >= one

        # Polynomial part for 0 < x < 1
        x_poly = x**3 * (ten - fifteen * x + six * x**2)

        return tf.where(cond0, zero, tf.where(cond1, one, x_poly))

    @tf.function
    def phi(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        x = tf.abs(x) / self.π#tf.constant(np.pi, dtype=tf.float64)

        cond1 = x <= 1
        cond2 = tf.logical_and(x > 1, x < 2)

        part1 = tf.ones_like(x)
        part2 = tf.cos(0.5 * self.π * self.v(x - 1))
        part3 = tf.zeros_like(x)

        return tf.where(cond1, part1, tf.where(cond2, part2, part3))
    
    @tf.function
    def W(self, x, ljbj, ljbj_1):
        return tf.math.sqrt(tf.math.square(self.phi(x/ljbj))-tf.math.square(self.phi(x/ljbj_1))) 

    @tf.function
    def _V_piecewise_fn(self, x):
        # x = tf.convert_to_tensor(x, dtype=tf.float32)
        cond = tf.logical_and(x > -1, x < 1)
        val = tf.sqrt(self.v(1.0 - tf.abs(x)))
        return tf.where(cond, val, tf.zeros_like(x))
    @tf.function
    def _V_wavedec(self, x):
        wavelet = pywt.Wavelet(self.wave)
        h0 = wavelet.dec_lo
        return self.freqz_abs(x, h0)
    @tf.function
    def _V_waverec(self, x):
        wavelet = pywt.Wavelet(self.wave)
        g0 = wavelet.rec_lo
        return self.freqz_abs(x, g0)
    @tf.function
    def freqz_abs(self, x1, h0):
        # Ensure high precision: float64 and complex128
        # h0 = tf.convert_to_tensor(h0, dtype=tf.complex128)
        x1 = self.π*x1
        
        h0 = h0 / tf.sqrt(tf.constant(2.0, dtype=tf.float64))
        h0 = tf.cast(h0, tf.complex128)

        x1 = tf.cast(x1, tf.float64)
        # π = tf.constant(np.pi, dtype=tf.float64)
        x1_clipped = tf.clip_by_value(x1, -self.π, self.π)
        # x1_clipped = tf.clip_by_value(x1, -tf.constant(tf.constant(np.pi, dtype=tf.float64)), tf.constant(np.pi, dtype=tf.float64))
        freqs = tf.reshape(x1_clipped, [-1])  # [M]

        n = tf.range(tf.shape(h0)[0], dtype=tf.float64)  # [L]
        n = tf.reshape(n, [1, -1])                      # [1, L]
        freqs = tf.reshape(freqs, [-1, 1])              # [M, 1]

        exponent = tf.exp(-1j * tf.cast(freqs * n, tf.complex128))  # [M, L]
        response = tf.matmul(exponent, tf.reshape(h0, [-1, 1]))     # [M, 1]
        response_abs = tf.abs(response)                             # [M, 1]

        return tf.cast(tf.reshape(response_abs, tf.shape(x1)), dtype=tf.float64)
        
    @tf.function
    def centered_wrap(self, x, half_modulo=np.pi):
        return (x+half_modulo)%(2*half_modulo)-half_modulo
    
    ## How to use this??
    @tf.function
    def downsample(self, x, fac=2):
        return tf.add_n(
            tf.split(
                tf.add_n(
                    tf.split(
                        tf.add_n(
                            tf.split(x, fac, axis=-1)
                        ), fac, axis=-2
                    )
                ), fac, axis=-3
            )
        )#.numpy()

    @tf.function
    def get_shearlet_system(self, filtertype='wavedec'):
        N = self.N
        L = self.L
        J = self.J
        B = self.B
        
        def V(x):
            if filtertype == "piecewise":
                return self._V_piecewise_fn(x)
            elif filtertype == "wavedec":
                return self._V_wavedec(x)
            elif filtertype == "waverec":
                return self._V_waverec(x)
            else:
                raise ValueError(f"Unknown mode: {filtertype}")

        shearlet_system = [[], [], [], []]
        temp = self.centered_wrap((tf.range(N, dtype=tf.float64) * (2.0 * self.π / N)), self.π)
        w = tf.stack(tf.meshgrid(temp, temp, temp, indexing='ij'), axis=0)
        
        temp = tf.math.abs(w)
        # print(f"{np.logical_and(temp[1]<=temp[0], temp[2]<=temp[0]).dtype} \n{np.logical_and(temp[1]<=temp[0], temp[2]<=temp[0]).dtype}")
        XP0 = tf.logical_and(temp[1]<=temp[0], temp[2]<=temp[0])#.numpy()
        XP1 = tf.logical_and(temp[0]<temp[1], temp[2]<=temp[1])#.numpy()
        XP2 = tf.logical_and(temp[0]<temp[2], temp[1]<temp[2])#.numpy()
        XP0 = tf.cast(XP0, dtype=tf.float64)
        XP1 = tf.cast(XP1, dtype=tf.float64)
        XP2 = tf.cast(XP2, dtype=tf.float64)

        # print(XP1.dtype,'//', XP1)


        w0 = w[0, :, :1, :1]
        shearlet = self.phi((L[J-1]*B[J-1])*w0)
        shearlet = shearlet*XP0 + tf.transpose(shearlet, perm=[1, 0, 2])*XP1 + tf.transpose(shearlet, perm=[1, 2, 0])*XP2
        shearlet_system[0].append(shearlet)
        
        temp = w[0, :, :, :1]
        # ratio = tf.math.divide(w[1, :, :, :1], temp, out=np.full_like(temp, np.inf), where=temp!=0)#.numpy()
        # ratio = np.divide(w[1, :, :, :1], temp, out=np.full_like(temp, np.inf), where=temp!=0) ## -----totf
        safe_temp = tf.where(temp != 0, temp, tf.constant(np.inf, dtype=temp.dtype))
        ratio = w[1, :, :, :1] / safe_temp


        for j in range(J):
            W0 = self.W(w0, (L[j]*B[j])/(L[J-1]*B[J-1]), (1 if j==0 else L[j-1]*B[j-1])/(L[J-1]*B[J-1]))
            W1 = tf.transpose(W0, perm=[1, 0, 2])*XP1
            W2 = tf.transpose(W0, perm=[1, 2, 0])*XP2
            W0 = W0*XP0
            temp = L[j]*ratio

            for l1 in range(-L[j], L[j]+1):
                shear1 = V(temp-l1)
                for l2 in range(-L[j], L[j]+1):
                    shearlet = shear1*tf.transpose(V(temp-l2), perm=[0, 2, 1])
                    if abs(l1)==L[j]:
                        shearlet_ = shear1*tf.transpose(V(temp+l2), perm=[0, 2, 1]) if l1<0 else shearlet
                        if abs(l2)==L[j]:
                            shearlet__ = V(temp+l1)*tf.transpose(V(temp-l2),perm=[0, 2, 1]) if l2<0 else shearlet
                            shearlet = shearlet*W0 + tf.transpose(shearlet_, perm=[1, 0, 2])*W1 + tf.transpose(shearlet__,perm=[2, 1, 0])*W2
                            shearlet_system[3].append(shearlet)
                        else:
                            shearlet = shearlet*W0 + tf.transpose(shearlet_, perm=[1, 0, 2])*W1
                            shearlet_system[2].append(shearlet)
                    elif abs(l2)!=L[j]:
                        shearlet = shearlet*W0
                        shearlet_system[1].append(shearlet)
        for i in range(4):
            shearlet_system[i] = tf.stack(shearlet_system[i], axis=0)
        print([shearlets.dtype for shearlets in shearlet_system])
        return shearlet_system

    def _yield_filters(self):
        yield self.shearlet_system[0]
        yield self.shearlet_system[1]
        yield tf.transpose(self.shearlet_system[1], perm=[0, 2, 3, 1])
        yield tf.transpose(self.shearlet_system[1], perm=[0, 3, 1, 2])
        yield self.shearlet_system[2]
        yield tf.transpose(self.shearlet_system[2], perm=[0, 2, 3, 1])
        yield tf.transpose(self.shearlet_system[2], perm=[0, 3, 1, 2])
        yield self.shearlet_system[3]
    def _yield_filter_bank_tensor(self):
        FB = [
            self.shearlet_system[0],
            self.shearlet_system[1],
            tf.transpose(self.shearlet_system[1], perm=[0, 2, 3, 1]),
            tf.transpose(self.shearlet_system[1], perm=[0, 3, 1, 2]),
            self.shearlet_system[2],
            tf.transpose(self.shearlet_system[2], perm=[0, 2, 3, 1]),
            tf.transpose(self.shearlet_system[2], perm=[0, 3, 1, 2]),
            self.shearlet_system[3]
        ]
        return tf.cast(tf.concat(FB, axis=0), tf.complex128)
    def _yield_filter_bank_tensor_rec(self):
        FB = [
            self.shearlet_system_rec[0],
            self.shearlet_system_rec[1],
            tf.transpose(self.shearlet_system_rec[1], perm=[0, 2, 3, 1]),
            tf.transpose(self.shearlet_system_rec[1], perm=[0, 3, 1, 2]),
            self.shearlet_system_rec[2],
            tf.transpose(self.shearlet_system_rec[2], perm=[0, 2, 3, 1]),
            tf.transpose(self.shearlet_system_rec[2], perm=[0, 3, 1, 2]),
            self.shearlet_system_rec[3]
        ]
        return tf.cast(tf.concat(FB, axis=0), tf.complex128)

    def forward(self, x):
        x = tf.cast(x, dtype=tf.complex128)
        xfft = tf.signal.fft3d(x)
        FB = self._yield_filter_bank_tensor()
        filtered = tf.einsum('ijk,fijk->fijk', xfft, FB)        
        filtered = tf.signal.ifft3d(filtered)
        # filtered = filtered[..., :x.shape[-3], :x.shape[-2], :x.shape[-1]]
        if self.norm:
            filtered *= self.norm_factors
        return filtered

    def call(self, x):
        if self.transform==None:
            return self.forward(x)
        elif self.transform=='inverse':
            return self.inverse(x)
        else:
            raise ValueError(f"Unknown key {transform}!! keys 'None' or 'inverse' only allowed")

    def inverse(self, y):
        y = tf.cast(y, tf.complex128)
        if self.norm:
            y = y/self.norm_factors      
        yfft = tf.signal.fft3d(y)
        if 'bio' in self.wave:
            FB = self._yield_filter_bank_tensor_rec()
        else:
            FB = self._yield_filter_bank_tensor()
        synthesized = tf.einsum('fijk,fijk->ijk', yfft, FB)
        # y = y*filters
        # synthesized = tf.cast(synthesized, tf.complex128)
        synthesized = tf.signal.ifft3d(synthesized)#, axes=(-3, -2, -1))
        # synthesized = synthesized[..., :y.shape[-3], :y.shape[-2], :y.shape[-1]]
        return tf.math.real(synthesized)
        # return synthesized

if __name__=='__main__':
    start_time = time.time()
    # ST3D = ShearletTransform3D(N=128, J=2, L=[1, 2], B=[4, 8], norm=True)
    # ST3D = ShearletTransform3D(N=32, J=2, L=[1, 2], B=[4, 8], norm=True, wave='db15')
    # ST3D = ShearletTransform3D(N=32, J=2, L=[1, 2], B=[4, 8], norm=True, wave='bior1.5')
    ST3D = ShearletTransform3D(N=128, J=2, L=[1, 2], B=[4, 8], norm=True, wave='db15')

    # ST3D = ShearletTransform3D(N=32, J=3, L=[1, 4, 8], B=[6, 8, 10], norm=True, wave='db20')
    print(time.time()-start_time)



    ## get the shearlet filters
    print(ST3D)
    ## get the shearlet system
    sys3d = ST3D.shearlet_system
    len(sys3d), type(sys3d)
    np.squeeze(sys3d[0]).shape
    # print filter shapes
    print('\nFilter shapes: ',[_.shape for _ in sys3d])
    print('Filter dtypes: ',[_.dtype for _ in sys3d])



    ## check perfect reconstruction
    n = 128
    axis1 = np.arange(0,n)
    x = np.einsum('i,j->ij', axis1, axis1)
    x = np.einsum('i,j,k->ijk', axis1, axis1, axis1)
    # viz(x)
    ST3D = ShearletTransform3D(N=n, J=2, L=[1, 2], B=[4, 8], norm=True, wave='rbio1.5')
    print(f"\nReconstruction error: {tf.reduce_sum(tf.abs(ST3D.inverse(ST3D.forward(x)) - x))}\n")
    # viz(ST3D.inverse(ST3D(x)))

    # s = np.random.randint(_[1].shape[0])
    # s=0 #override!!
    # print(s)
    # _ = _[1][2,...]
    # viz(np.squeeze()), plt.title(s)`