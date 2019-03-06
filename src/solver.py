from math import pi

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tnrange

EPS = 1e-8

def diff_x_up(psi):
    return psi[1:-1] - psi[:-2]

def diff_x_down(psi):
    return psi[2:] - psi[1:-1]

# 2nd order flux using combination of F_L (upwind) and F_H (Lax-Wendroff) and van_leer_limiter (can be changed to others).
def van_leer_limiter(r):
    return (r + np.abs(r))/(1. + np.abs(r))

# flux for upwind direction
def F_p_2(f, vp, dx, dt, limiter=van_leer_limiter):
    r = (f[1:-1] - f[:-2])/(f[2:] - f[1:-1] + EPS)
    phi = limiter(r)
    F = f.copy()
    F[1:-1] += 0.5*phi*(1.- vp*dt/dx)*(f[2:] - f[1:-1])
    return (F[2:-2] - F[1:-3])/dx

# flux for downwind direction
def F_m_2(f, vm, dx, dt, limiter=van_leer_limiter):
    r = (f[2:] - f[1:-1])/(f[1:-1] - f[:-2] + EPS)
    phi = limiter(r)
    F = f.copy()
    F[1:-1] += 0.5*phi*(-1.- vm*dt/dx)*(f[1:-1] - f[:-2])
    return (F[3:-1] - F[2:-2])/dx

def grad_vf2(state, dt):
    dx = state.delta[0]
    vp = np.fmax(state.c_centers[-2], 0.)
    vm = np.fmin(state.c_centers[-2], 0.)
    # v*grad_f
    return vp[2:-2]*F_p_2(state.f,vp[1:-1],dx,dt) + vm[2:-2]*F_m_2(state.f,vm[1:-1],dx,dt)

def grad_vf1(state):
    dx = state.delta[0]
    v = state.c_centers[-2]
    vp = 0.5*(v + np.abs(v))
    vm = 0.5*(v - np.abs(v))
    return diff_x_up(vp*state.f)/dx + diff_x_down(vm*state.f)/dx



class BoltzmannSolver(object):

    def __init__(self, kn, riemann_solver, collision_operator, stepper='proj_int'):
        self.convection = riemann_solver
        self.collision = collision_operator
        self.kn = kn
        # step
        self.set_time_stepper(stepper)

    def dfdt(self, state):
        return -self.convection(state) + 1/self.kn*self.collision(state.f)[1:-1]

    def euler(self, solution, dt):
        solution.state.f[1:-1] = solution.state.f[1:-1] + dt*self.dfdt(solution.state)

    def rk4(self, solution, dt):
        state = solution.state
        f0 = state.f.copy()[1:-1]
        k1 = self.dfdt(state)
        state.f[1:-1] = f0 + 0.5*k1
        k2 = self.dfdt(state)
        state.f[1:-1] = f0 + 0.5*k2
        k3 = self.dfdt(state)
        state.f[1:-1] = f0 + 0.5*k3
        k4 = self.dfdt(state)

        state.f[1:-1] = f0 + dt*(1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4)

    def pfe(self, solution, Dt, dt, nt):
        state = solution.state

        for _ in tnrange(nt, desc='Inner', leave=False):
            self.euler(solution, dt)
        state.f[1:-1] = state.f[1:-1] + (Dt - nt*dt)*self.dfdt(state)

    def prk4(self, solution, Dt, dt, nt):
        state = solution.state

        for _ in tnrange(nt, desc='k1', leave=False):
            self.euler(solution, dt)
        k1 = self.dfdt(solution.state)
        f_nt_1 = (dt*k1 + state.f[1:-1]).copy()

        state.f[1:-1] = f_nt_1 + (0.5*Dt - (nt+1)*dt)*k1
        for _ in tnrange(nt, desc='k2', leave=False):
            self.euler(solution, dt)
        k2 = self.dfdt(state)

        state.f[1:-1] = f_nt_1 + (0.5*Dt - (nt+1)*dt)*k2
        for _ in tnrange(nt, desc='k3', leave=False):
            self.euler(solution, dt)
        k3 = self.dfdt(state)

        state.f[1:-1] = f_nt_1 + (Dt - (nt+1)*dt)*k3
        for _ in tnrange(nt, desc='k4', leave=False):
            self.euler(solution, dt)
        k4 = self.dfdt(state)

        state.f[1:-1] = f_nt_1 + (Dt - (nt+1)*dt)*(1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4)

    def set_time_stepper(self, stepper='euler'):
        """Sets the time step scheme to be used while solving given a
        string which should be one of ['euler', 'proj_int']."""
        if stepper == 'euler':
            self.step = lambda solution, Dt, dt, nt: self.euler(solution, Dt)
        elif stepper == 'pfe':
            self.step = self.pfe
        elif stepper == 'prk4':
            self.step = self.prk4

    def solve(self, solution, Dt, Nt, dt, nt):
        for _ in tnrange(Nt, desc='Outer'):
            self.step(solution, Dt, dt, nt)
            solution.state.t += Dt
