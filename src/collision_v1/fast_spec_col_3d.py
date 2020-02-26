# fast_spec_col_3d

from math import pi

import numpy as np
import cupy as cp
import pyfftw

EPS = 1e-8


def sinc(x):
    return np.sin(x+EPS)/(x+EPS)


class FastSpectralCollison3D(object):
    def __init__(self, config, e=None, N=None, GPU=True):
        self._config = config
        self._gamma = config.physical_config.gamma
        if e is None:
            self._e = config.physical_config.e
        else:
            self._e = e

        S = config.domain_config.S
        self._R = 2*S
        self._L = eval(config.domain_config.L)
        if N is None:
            self._N = config.domain_config.N
        else:
            self._N = N

        self._N_R = config.quadrature_config.N_g
        self._method = config.quadrature_config.method

        self._dv = None
        self._v = None
        self._v_norm = None

        self._precompute()
        self._fftw_plan()
        if GPU:
            self._init_gpu_arrays()

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
            self._v_norm = (self.v**2)[:, None, None] + \
                (self.v**2)[:, None] + self.v**2
        return self._v_norm

    @property
    def N(self):
        return self._N

    @property
    def e(self):
        return self._e

    @property
    def fft3(self):
        return self._fft3

    @property
    def ifft3(self):
        return self._ifft3

    def col_full(self, f):
        # fft of f
        f_hat = self._fft3(f)
        # convolution and quadrature
        Q = np.sum(self._r_w*self._s_w[:, None]*(self._F_k_gain - self._F_k_loss)
                            * self._fft5(self._ifft5(self._exp*f_hat[..., None, None])*f[..., None, None]), axis=(-1, -2))
        return np.real(self._ifft3(Q))

    def col_sep(self, f):
        # fft of f
        f_hat = self._fft3(f)
        # gain term
        Q_gain = np.sum(self._s_w[:, None]*self._r_w*self._F_k_gain
                        * self._fft5(self._ifft5(self._exp*f_hat[..., None, None])*f[..., None, None]), axis=(-1, -2))
        # loss term
        Q_loss = 4*pi*np.sum(self._r_w*self._F_k_loss
                             * self._fft4(self._ifft4(self._sinc*f_hat[..., None])*f[..., None]), axis=(-1))
        return np.real(self._ifft3(Q_gain - Q_loss))

    def laplacian(self, f):
        return np.real(self._ifft3(self._lapl*self._fft3(f)))

    def col_heat_full(self, f_hat, eps):
        # ifft
        f = self._ifft3(f_hat)
        # convolution and quadrature
        Q = np.sum(self._s_w[:, None]*self._r_w*(self._F_k_gain - self._F_k_loss)
                   * self._fft5(self._ifft5(self._exp*f_hat[..., None, None])*f[..., None, None]), axis=(-1, -2))
        return Q/(4*pi) + eps*self._lapl*f_hat

    def col_heat_sep(self, f_hat, eps):
        # ifft of f
        f = self._ifft3(f_hat)
        # gain term
        Q_gain = np.sum(self._s_w[:, None]*self._r_w*self._F_k_gain
                        * self._fft5(self._ifft5(self._exp*f_hat[..., None, None])*f[..., None, None]), axis=(-1, -2))
        # loss term
        Q_loss = 4*pi*np.sum(self._r_w*self._F_k_loss
                             * self._fft4(self._ifft4(self._sinc*f_hat[..., None])*f[..., None]), axis=(-1))
        return (Q_gain - Q_loss)/(4*pi) + eps*self._lapl*f_hat

    def col_gpu(self, f):
        f_gpu = cp.asarray(f)
        # fft of f
        f_hat = cp.fft.fftn(f_gpu, axes=(0,1,2))
        # gain term
        Q_gain = cp.sum(self._s_w[:, None]*self._r_w*self._F_k_gain
                        * cp.fft.fftn(cp.fft.ifftn(self._exp*f_hat[..., None, None], axes=(0,1,2))*f_gpu[..., None, None], axes=(0,1,2)), axis=(-1,-2))
        # loss term
        Q_loss = 4*pi*cp.sum(self._r_w*self._F_k_loss
                             * cp.fft.fftn(cp.fft.ifftn(self._sinc*f_hat[..., None], axes=(0,1,2))*f_gpu[..., None], axes=(0,1,2)), axis=-1)
        return cp.real(cp.fft.ifftn(Q_gain - Q_loss, axes=(0,1,2)))
    
    def _init_gpu_arrays(self):
        self._s_w_gpu = cp.asarray(self._s_w)
        self._r_w_gpu = cp.asarray(self._r_w)
        self._F_k_gain_gpu = cp.asarray(self._F_k_gain)
        self._exp_gpu = cp.asarray(self._exp)
        self._F_k_loss_gpu = cp.asarray(self._F_k_loss)
        self._sinc_gpu = cp.asarray(self._sinc)

    def _precompute(self):
        # legendre quadrature
        x, w = np.polynomial.legendre.leggauss(self._N_R)
        r = 0.5*(x + 1)*self._R
        self._r_w = 0.5*self._R*w
        if self._method == "spherical_design":
            # spherical points and weight
            self._sigma = np.loadtxt(self._config.quadrature_config.sigma)
            self._s_w = 4*pi/self._sigma.shape[0]*np.ones(self._sigma.shape[0])
        else:
            self._N_theta = self._config.quadrature_config.N_theta
            self._N_phi = self._config.quadrature_config.N_phi
            # integral on unit sphere using quadrature
            theta, w_theta = np.polynomial.legendre.leggauss(self._N_theta)
            theta = 0.5*(theta + 1)*pi
            w_theta = 0.5*pi*w_theta
            phi, w_phi = np.polynomial.legendre.leggauss(self._N_phi)
            phi = 0.5*(phi + 1)*2*pi
            w_phi = 0.5*2*pi*w_phi
            self._sigma = np.zeros((self._N_theta*self._N_phi, 3))
            self._sigma[:, 0] = (np.sin(theta)[:, None]*np.cos(phi)).flatten()
            self._sigma[:, 1] = (np.sin(theta)[:, None]*np.sin(phi)).flatten()
            self._sigma[:, 2] = (np.cos(theta)[:, None] *
                                 np.ones(self._N_phi)).flatten()
            self._s_w = ((np.sin(theta)*w_theta)[:, None]*w_phi).flatten()
        # index
        k = np.fft.fftshift(np.arange(-self._N/2, self._N/2))
        # dot with index
        rkg = (k[:, None, None, None]*self._sigma[:, 0]
               + k[:, None, None]*self._sigma[:, 1]
               + k[:, None]*self._sigma[:, 2])[..., None]*r
        # norm of index
        k_norm = np.sqrt(k[:, None, None]**2 + k[:, None]**2 + k**2)
        # gain kernel
        self._F_k_gain = 4*pi*r**(self._gamma+2)*(np.exp(0.25*1j*pi*(1+self._e)*rkg/self._L)
                                                  * sinc(0.25*pi*(1+self._e)*r*k_norm[..., None, None]/self._L))
        # loss kernel
        self._F_k_loss = 4*pi*r**(self._gamma+2)
        # exp for fft
        self._exp = np.exp(-1j*pi*rkg/self._L)
        # sinc
        self._sinc = sinc(pi*r*k_norm[..., None]/self._L)
        # laplacian
        self._lapl = -pi**2/self._L**2*k_norm**2

    def _fftw_plan(self, num_thread=8):
        N, N_R = self._N, self._N_R
        # pyfftw planning of (N, N, N)
        array_3d = pyfftw.empty_aligned((N, N, N), dtype='complex128')
        self._fft3 = pyfftw.builders.fftn(array_3d, overwrite_input=True,
                                          planner_effort='FFTW_ESTIMATE', threads=8, avoid_copy=True)
        self._ifft3 = pyfftw.builders.ifftn(array_3d, overwrite_input=True,
                                            planner_effort='FFTW_ESTIMATE', threads=8, avoid_copy=True)
        # pyfftw planning of (N, N, N, N_R)
        array_4d = pyfftw.empty_aligned((N, N, N, N_R), dtype='complex128')
        self._fft4 = pyfftw.builders.fftn(array_4d, axes=(0, 1, 2), overwrite_input=True,
                                          planner_effort='FFTW_ESTIMATE', threads=8, avoid_copy=True)
        self._ifft4 = pyfftw.builders.ifftn(array_4d, axes=(0, 1, 2), overwrite_input=True,
                                            planner_effort='FFTW_ESTIMATE', threads=8, avoid_copy=True)
        # pyfftw planning of (N, N, N, sigma, N_R)
        array_5d = pyfftw.empty_aligned(
            (N, N, N, self._sigma.shape[0], N_R), dtype='complex128')
        self._fft5 = pyfftw.builders.fftn(array_5d, axes=(0, 1, 2), overwrite_input=True,
                                          planner_effort='FFTW_ESTIMATE', threads=8, avoid_copy=True)
        self._ifft5 = pyfftw.builders.ifftn(array_5d, axes=(0, 1, 2), overwrite_input=True,
                                            planner_effort='FFTW_ESTIMATE', threads=8, avoid_copy=True)
