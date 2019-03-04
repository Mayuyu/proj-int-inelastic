from math import pi

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tnrange
from fast_spec_col_2d import FastSpectralCollison2D
from utility import *


class BoltzmannSolver(object):
    """A iterative Boltzmann solver that can use different schemes."""

    def __init__(self, kn, stepper='proj_int'):

        self.kn = kn        
        # step
        self.set_time_stepper(stepper)
 
    def dq_euler(self, state):
        return - grad_vf(state) + 1/self.kn*state.Q.col_sep(state.f)[1:-1]

    def euler(self, state, Dt, dt, nt):
        state.f[1:-1] += Dt*self.dq_euler(state)

    def proj_int(self, state, Dt, dt, nt):
        for _ in tnrange(nt, desc='Inner', leave=False):
            state.f[1:-1] += dt*self.dq_euler(state)
        state.f[1:-1] += (Dt - nt*dt)*self.dq_euler(state)
    
    def set_time_stepper(self, stepper='euler'):
        """Sets the time step scheme to be used while solving given a
        string which should be one of ['euler', 'proj_int']."""
        if stepper == 'euler':
            self.time_step = self.euler
        elif stepper == 'proj_int':
            self.time_step = self.proj_int
   
    def solve(self, state, Dt, Nt, dt, nt):
        for _ in tnrange(Nt, desc='Outer'):
            self.time_step(state, Dt, dt, nt)


class State(object):
    """A grid class that stores the details and solution of the
    computational grid."""

    def __init__(self, config):
        self.xmin, self.xmax = config.xmin, config.xmax
        S = config.s
        L = eval(config.lv)
        self.vmin, self.vmax = -L, L
        self.nx, self.nv = config.nx, config.nv
        self.dx = dx = float(self.xmax - self.xmin)/config.nx
        self.dv = dv = float(self.vmax - self.vmin)/config.nv
        self.x = np.arange(self.xmin, self.xmax+dx, dx)
        self.v = np.arange(self.vmin+dv/2, self.vmax+dv/2, dv)
        # v+, v-.
        abs_v = np.abs(self.v)
        self.vp = 0.5*(self.v + abs_v)
        self.vm = 0.5*(self.v - abs_v)

        self.Q = FastSpectralCollison2D(config)
        # Store the probability density function.
        self.f = np.zeros(self.x.shape+(self.nv, self.nv))
        # used to compute the change in solution in some of the methods.
        self.old_rho = self.density().copy()

    def set_initial(self, f0):
        self.f = f0.copy()

    def set_bc_func(self, func_l, func_r):
        """Sets the BC given a function of two variables."""
        self.f[0, self.v > 0] = func_l(self.v)[self.v > 0]
        self.f[-1, self.v < 0] = func_r(self.v)[self.v < 0]

    def density(self):
        """Compute the macro quantity: density."""
        return np.sum(self.f, axis=(-1, -2))*self.dv**2

    def velocity(self):
        return np.sum(self.f*self.v[:, None], axis=(-1, -2))*self.dv**2/self.density()

    def temperature(self):
        return 0.5*np.sum(self.f*(self.v[:, None]**2 + self.v**2), axis=(-1, -2))*self.dv**2

    def compute_error(self):
        """Computes absolute error using an L2 norm for the solution.
        This requires that self.f and self.old_f must be appropriately
        setup."""
        v = (self.density() - self.old_rho).flat
        return np.sqrt(np.dot(v, v)*self.dx)

    def plot(self, i, cl=1):
        """Plot the contour in velocity space for given index i"""
        dv = self.dv
        v = np.mgrid[self.vmin+dv/2:self.vmax+dv/2:dv,
                     self.vmin+dv/2:self.vmax+dv/2:dv]

        fig, ax = plt.subplots()
        cs = ax.contour(v[0], v[1], self.f[i])
        if cl == 1:
            ax.clabel(cs, inline=0.5)

        ax.grid(linestyle=':')
        plt.show()

    def plot_macro(self, macro='velocity'):
        """Plot the macroscopic quantities: density, velocity and temperature."""
        fig, ax = plt.subplots()
        if macro == 'density':
            cs = ax.plot(self.x, self.density())
            ax.set_ylabel(r'$\rho(x)$', fontsize='large')
        elif macro == 'temperature':
            cs = ax.plot(self.x, self.temperature())
            ax.set_ylabel(r'$T(x)$', fontsize='large')
        else:
            cs = ax.plot(self.x, self.velocity())
            ax.set_ylabel(r'$u(x)$', fontsize='large')

        ax.set_xlabel(r'$x$', fontsize='large')
        ax.grid(which='both', linestyle=':')
        plt.show()
