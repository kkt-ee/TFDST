import tensorflow as tf
import numpy as np
from TFDST.dbFBimpulseResponse import FBimpulseResponses
# from scipy.signal import freqz
import time


class ShearletTransform2D(tf.keras.layers.Layer):
    """Shearlet transform 2D base layer.
    
    TFDST: Fast Discrete Shearlet Transform Layers in TensorFlow.
    Copyright 2025 Kishore Kumar Tarafdar.
    Licensed under the Apache License, Version 2.0. See LICENSE for details.

    Filter-bank construction follows mathematical structure developed with
    credit to Vineet Ghule. This implementation provides differentiable
    TensorFlow/Keras layers with batched multichannel support and
    performance-oriented updates.
    """
    def __init__(
        self, N, 
        J, 
        L=None, B=None, 
        a=None, 
        norm=False, 
        wave='db3',
        real_coefficients=True,
        transform=None, ## inverse
        **kwargs):
        kwargs.setdefault('autocast', False)
        super().__init__(**kwargs)
        self.transform = transform ## inverse
        self.wave = wave
        self.π = tf.constant(np.pi, dtype=tf.float64)
        self.N = N
        self.J = J
        self.a = a
        if a is not None:
            L = [int(tf.math.round(a**(0.5*j))) for j in range(J)]
            B = [a**(j+1)/L[j] for j in range(J)]
        self.L = L
        self.B = B
        self.norm = norm
        self.real_coefficients = real_coefficients
        self.shearlet_system = self.get_shearlet_system('wavedec')
        if 'bio' in self.wave:
            self.shearlet_system_rec = self.get_shearlet_system('waverec')
        else: self.shearlet_system_rec = None
        if self.norm:
            self.norm_factors = self.get_norm_factors()
        
    @tf.function
    def get_norm_factors(self):
        """ Normalizing with thre reconstruction Filterbank!!! recheck here!!!"""       
        FBdec, _ = self.getFB()
        # Compute the squared L2 norm over the spatial dimensions
        norm_factors = tf.reduce_sum(tf.math.square(tf.math.abs(FBdec)), axis=(-2, -1), keepdims=True)  # shape: [F, 1, 1, 1]
        norm_factors = tf.math.sqrt(self.N**2/norm_factors)
        return tf.cast(norm_factors, tf.complex128)
         
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
        # Conditions
        cond1 = x <= 1
        cond2 = tf.logical_and(x > 1, x < 2)
        # phi parts
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
        h0 = FBimpulseResponses[self.wave][0][0][::-1]
        return self.freqz_abs(x, h0)
    @tf.function
    def _V_waverec(self, x):
        g0 = FBimpulseResponses[self.wave][1][0]
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

    # ## How to use this??
    @tf.function
    def downsample(self, x, fac=2):
        return tf.add_n(tf.split(tf.add_n(tf.split(x, fac, axis=-1)), fac, axis=-2))#.numpy()

    # @tf.function
    def get_shearlet_system(self, filtertype):

        def V(x):
            if filtertype == "piecewise":
                return self._V_piecewise_fn(x)
            elif filtertype == "wavedec":
                return self._V_wavedec(x)
            elif filtertype == "waverec":
                return self._V_waverec(x)
            else:
                raise ValueError(f"Unknown mode: {filtertype}")

        N = self.N
        J = self.J
        L = self.L
        B = self.B
        shearlet_system = [[], [], []]
        temp = self.centered_wrap((tf.range(N, dtype=tf.float64) * (2.0 * self.π / N)), self.π)
        w = tf.stack(tf.meshgrid(temp, temp, indexing='ij'), axis=0)
        
        temp = tf.math.abs(w)
        XP0 = temp[1]<=temp[0]
        XP1 = tf.logical_not(XP0)
        XP0 = tf.cast(XP0, dtype=tf.float64)
        XP1 = tf.cast(XP1, dtype=tf.float64)

        w0 = w[0, :, :1]
        shearlet = self.phi((L[J-1]*B[J-1])*w0)
        shearlet = shearlet*XP0 + tf.transpose(shearlet, perm=[1, 0])*XP1
        shearlet_system[0].append(shearlet)
        
        temp = w[0]
        # ratio = np.divide(w[1], temp, out=np.full_like(temp, np.inf), where=temp!=0)
        safe_temp = tf.where(temp != 0, temp, tf.constant(1.0, dtype=temp.dtype))
        ratio = tf.where(temp != 0, w[1] / safe_temp, tf.constant(float('inf'), dtype=temp.dtype))
        # safe_temp = tf.where(temp != 0, temp, tf.constant(np.inf, dtype=temp.dtype))
        # ratio = w[1] / safe_temp

        for j in range(J):
            W0 = self.W(w0, (L[j]*B[j])/(L[J-1]*B[J-1]), (1 if j==0 else L[j-1]*B[j-1])/(L[J-1]*B[J-1]))
            W1 = tf.transpose(W0, perm=[1, 0])*XP1
            W0 = W0*XP0
            temp = L[j]*ratio

            for l1 in range(-L[j], L[j]+1):
                shearlet = V(temp-l1)
                if abs(l1)==L[j]:
                    shearlet = shearlet*W0 + tf.transpose(shearlet, perm=[1, 0])*W1
                    shearlet_system[2].append(shearlet)
                else:
                    shearlet = shearlet*W0
                    shearlet_system[1].append(shearlet)
        for i in range(3):
            shearlet_system[i] = tf.stack(shearlet_system[i], axis=0)
        return shearlet_system

    # UPDATE
    def getFB(self):
        if 'bio' in self.wave:
            return self._analysis_bank_tensor(), self._synthesis_bank_tensor_biortho()
        else:
            return self._analysis_bank_tensor(), None

    def _nyquist_real_correction(self, FB):
        """Apply the FFST even-size correction for real coefficients.

        ``FB`` uses TensorFlow's unshifted FFT ordering.  The Nyquist row and
        column are therefore at ``N // 2`` rather than at index zero as in the
        shifted-grid formulation in Section 3.8.1 of the FFST paper.
        """
        if not self.real_coefficients or self.N % 2:
            return FB

        # Filter-bank layout: lowpass, all horizontal interior shears, all
        # vertical interior shears, then two seam filters per scale.  FFST
        # corrects only the non-axis-aligned filters at the finest scale.
        interior_counts = [2 * int(l) - 1 for l in self.L]
        preceding = sum(interior_counts[:-1])
        total = sum(interior_counts)
        finest_l = int(self.L[-1])

        horizontal_start = 1 + preceding
        vertical_start = 1 + total + preceding
        shear_offsets = [
            i for i, shear in enumerate(range(-finest_l + 1, finest_l))
            if shear != 0
        ]
        filter_indices = (
            [horizontal_start + i for i in shear_offsets]
            + [vertical_start + i for i in shear_offsets]
            + [1 + 2 * total + 2 * (self.J - 1) + i for i in range(2)]
        )

        n = self.N
        nyquist = n // 2
        reverse_indices = tf.math.floormod(-tf.range(n), n)
        filter_mask = tf.reduce_any(
            tf.equal(
                tf.range(tf.shape(FB)[0])[:, tf.newaxis],
                tf.constant(filter_indices, dtype=tf.int32)[tf.newaxis, :],
            ),
            axis=1,
        )[:, tf.newaxis, tf.newaxis]

        coordinates = tf.range(n)
        row_mask = tf.logical_and(
            coordinates[:, tf.newaxis] == nyquist,
            coordinates[tf.newaxis, :] != nyquist,
        )[tf.newaxis, :, :]
        column_mask = tf.logical_and(
            coordinates[:, tf.newaxis] != nyquist,
            coordinates[tf.newaxis, :] == nyquist,
        )[tf.newaxis, :, :]
        scale = tf.math.rsqrt(tf.cast(2.0, FB.dtype))
        mirrored_row = scale * (FB + tf.gather(FB, reverse_indices, axis=2))
        mirrored_column = scale * (FB + tf.gather(FB, reverse_indices, axis=1))
        corrected = tf.where(tf.logical_and(filter_mask, row_mask), mirrored_row, FB)
        return tf.where(
            tf.logical_and(filter_mask, column_mask), mirrored_column, corrected
        )

    def _analysis_bank_tensor(self):
        FB = [
            self.shearlet_system[0],
            self.shearlet_system[1],
            tf.transpose(self.shearlet_system[1], perm=[0, 2, 1]),
            self.shearlet_system[2],
           ]
        FB = self._nyquist_real_correction(tf.concat(FB, axis=0))
        return tf.cast(FB, tf.complex128)
    def _synthesis_bank_tensor_biortho(self):
        FB = [
            self.shearlet_system_rec[0],
            self.shearlet_system_rec[1],
            tf.transpose(self.shearlet_system_rec[1], perm=[0, 2, 1]),
            self.shearlet_system_rec[2],
        ]
        FB = self._nyquist_real_correction(tf.concat(FB, axis=0))
        return tf.cast(FB, tf.complex128)

    def forward(self, x):
        x = tf.convert_to_tensor(x)
        input_is_real = not x.dtype.is_complex
        x = tf.cast(x, dtype=tf.complex128)
        xfft = tf.signal.fft2d(x)
        # FB = self._analysis_bank_tensor()
        FB, _ = self.getFB()
        filtered = tf.einsum('ij,fij->fij', xfft, FB)        
        filtered = tf.signal.ifft2d(filtered)
        # filtered = filtered[..., :x.shape[-3], :x.shape[-2], :x.shape[-1]]
        if self.norm:
            filtered *= self.norm_factors
        if self.real_coefficients and input_is_real:
            return tf.math.real(filtered)
        return filtered

    def call(self, x):
        if self.transform==None:
            return self.forward(x)
        elif self.transform=='inverse':
            return self.inverse(x)
        else:
            raise ValueError(f"Unknown key {self.transform}!! keys 'DST' or 'IDST' only allowed")

    def inverse(self, y):
        y = tf.cast(y, tf.complex128)
        if self.norm:
            y = y/self.norm_factors
        yfft = tf.signal.fft2d(y)             
        if 'bio' in self.wave:
            _, FB = self.getFB()
        else:
            FB, _ = self.getFB()        
        synthesized = tf.einsum('fij,fij->ij', yfft, FB)
        synthesized = tf.signal.ifft2d(synthesized)#, axes=(-3, -2, -1))
        return tf.math.real(synthesized)
        # return synthesized

    def get_config(self): 
        config = super().get_config()
        config.update({
            "N": self.N,
            "J": self.J,
            "L": self.L,
            "B": self.B,
            "a": self.a,
            "wave": self.wave,
            "norm": self.norm,
            "real_coefficients": self.real_coefficients,
            "transform": self.transform,
            # "shearlet_system": self.shearlet_system,
            # "shearlet_system_rec": self.shearlet_system_rec
            # Optional: include any other parameters used in `get_shearlet_system`
            # and `get_norm_factors`, if needed for a full reconstruction
            })
        return config

if __name__=='__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
    n = 128
    start_time = time.time()
    # ST2D = ShearletTransform2D(N=n, J=3, L=[2, 4, 8], B=[1, 1, 1], norm=True)
    ST2D = ShearletTransform2D(N=n, J=3, L=[2, 4, 8], B=[1, 1, 1], norm=True, wave='db25')
    ST2D = ShearletTransform2D(N=n, J=3, L=[2, 4, 8], B=[1, 1, 1], norm=True, wave='rbio1.5')
    print('elapsed time', time.time()-start_time)

    x = np.random.randn(n, n)
    start_time = time.time()
    y = ST2D.forward(x)
    print('elapsed time DST', time.time()-start_time)
    print(y.shape)

    start_time = time.time()
    x_ = ST2D.inverse(y)
    print('elapsed time IDST',time.time()-start_time)
    print(x_.shape)

    print('max. reconstruction error ', np.max(np.abs(x_-x)))
