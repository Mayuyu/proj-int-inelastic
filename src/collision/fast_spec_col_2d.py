# fast_spec_col_2d

import multiprocessing
from math import pi

import numpy as np
import pyfftw
from scipy import special

pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'

DTYPE = 'complex128'


class FastSpectralCollision2D(object):

    # Initialize parameters
    def __init__(self, config, e=None, N=None):
        # import parameters from config file
        self._gamma = config.gamma
        if e is None:
            self._e = config.e
        else:
            self._e = e

        s = config.s
        self._R = 2*s
        self._L = 0.5*(3 + np.sqrt(2))*s
#         self._L = config.l
        if N is None:
            self._N = config.nv
        else:
            self._N = N

        self._N_R = config.nv_g
        self._M = config.nv_sigma

        self._dv = None
        self._v = None
        self._v_norm = None

        self.num_dim_x = config.dim_x
        if self.num_dim_x:
            self._nx = config.nx

        self._fftw_planning()
        self._precompute()

    def col_full(self, f):
        # fft of f
        f_hat = self._fft2(f)
        # convolution and quadrature
        Q = self._s_w * np.sum(
            self._r_w * (self._F_k_gain-self._F_k_loss)
            * self._fft4(self._ifft4(self._exp * f_hat[..., None, None])
                         * f[..., None, None]),
            axis=(-1, -2)
        )
        return np.real(self._ifft2(Q)) / (2*pi)

    def col_sep(self, f):
        # fft of f
        f_hat = self._fft2(f)
        # gain term
        Q_gain = self._s_w * np.sum(
            self._r_w * self._F_k_gain
            * self._fft4(self._ifft4(
                self._exp*f_hat[..., None, None])
                * f[..., None, None]),
            axis=(-1, -2)
        )
        # loss term
        Q_loss = 2 * pi * np.sum(
            self._r_w * self._F_k_loss
            * self._ifft3(self._j0*f_hat[..., None])
            * f[..., None],
            axis=(-1)
        )
        return np.real(self._ifft2(Q_gain) - Q_loss) / (2*pi)

    def col_new(self, f):
        f_hat = self._fft2(f).copy()
        Q_gain, Q_loss = 0., 0.
        for j in range(self._N_R):
            Q_loss += 2 * pi * self._r_w[j] * self._F_k_loss[j] \
                * self._ifft2(self._j0[:, :, j] * f_hat) * f
            for i in range(self._M):
                Q_gain += self._s_w * self._r_w[j] * self._F_k_gain[:, :, i, j] \
                    * self._fft2(self._ifft2(self._exp[:, :, i, j] * f_hat) * f)

        return np.real(self._ifft2(Q_gain) - Q_loss) / (2*pi)

    def laplacian(self, f):
        return np.real(self._ifft2(self._lapl*self._fft2(f)))

    def col_heat(self, f, eps, col='new'):
        if col == 'new':
            col = self.col_new
        return col(f) + eps * self.laplacian(f)

    # ========================================
    # Precompute quantities and FFTw planning
    # ========================================

    def _precompute(self):
        # legendre quadrature
        x, w = np.polynomial.legendre.leggauss(self._N_R)
        r = 0.5 * (x+1) * self._R
        self._r_w = 0.5 * self._R * w
        # circular points and weight
        self._s_w = 2 * pi / self._M
        m = np.arange(0, 2*pi, self._s_w)
        # index
        k = np.fft.fftshift(np.arange(-int(self._N/2), int(self._N/2)))
        # dot with index
        rkg = (k[:, None, None]*np.cos(m) +
               k[:, None]*np.sin(m))[..., None] * r
        # norm of index
        k_norm = np.sqrt(k[:, None]**2 + k**2)
        # gain kernel
        self._F_k_gain = 2 * pi * r**(self._gamma+1) \
            * np.exp(0.25 * 1j * pi * (1+self._e) * rkg / self._L) \
            * special.jv(0, 0.25 * pi * (1+self._e) * r * k_norm[..., None, None] / self._L)
        # loss kernel
        self._F_k_loss = 2 * pi * r**(self._gamma+1)
        # exp for fft
        self._exp = np.exp(-1j * pi * rkg / self._L)
        # j0
        self._j0 = special.jv(0, pi * r * k_norm[..., None] / self._L)
        # laplacian
        self._lapl = -pi**2 / self._L**2 * k_norm**2

    def _fftw_planning(self):
        N, M, N_R = self._N, self._M, self._N_R
        if self.num_dim_x == 0:
            shape = (N, N)
        elif self.num_dim_x == 1:
            shape = (self._nx, N, N)
        elif self.num_dim_x == 2:
            shape = (self._nx, self._nx, N, N)
        else:
            raise Exception("x dim should be less than v dim.")

        # pyfftw planning of (N, N)
        array_2d = pyfftw.empty_aligned(shape, dtype=DTYPE)
        self._fft2 = pyfftw.builders.fft2(
            array_2d,  overwrite_input=True, avoid_copy=True)
        self._ifft2 = pyfftw.builders.ifft2(
            array_2d, overwrite_input=True, avoid_copy=True)
        # pyfftw planning of (N, N, N_R)
        array_3d = pyfftw.empty_aligned(shape+(N_R,), dtype=DTYPE)
        self._fft3 = pyfftw.builders.fftn(array_3d, axes=(-2, -3))
        self._ifft3 = pyfftw.builders.ifftn(array_3d, axes=(-2, -3))
        # pyfftw planning of (N, N, M, N_R)
        array_4d = pyfftw.empty_aligned(shape+(M, N_R), dtype=DTYPE)
        self._fft4 = pyfftw.builders.fftn(array_4d, axes=(-3, -4))
        self._ifft4 = pyfftw.builders.ifftn(array_4d, axes=(-3, -4))

    # =============================================
    # Attributes
    # =============================================

    @property
    def dv(self):
        if self._dv is None:
            self._dv = 2*self._L/self._N
        return self._dv

    @property
    def v(self):
        if self._v is None:
            self._v = np.arange(-self._L + self.dv/2,
                                self._L+self.dv/2, self.dv)
        return self._v

    @property
    def v_norm(self):
        if self._v_norm is None:
            self._v_norm = (self.v**2)[:, None] + self.v**2
        return self._v_norm

    @property
    def N(self):
        return self._N

    @property
    def e(self):
        return self._e

    @property
    def fft2(self):
        return self._fft2

    @property
    def ifft2(self):
        return self._ifft2
