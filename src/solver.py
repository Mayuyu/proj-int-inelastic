from math import pi

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tnrange


def diff_x_up(psi):
    return psi[1:-1] - psi[:-2]


def diff_x_down(psi):
    return psi[2:] - psi[1:-1]


def grad_vf(state):
    vp = np.fmax(state.c_centers[-2], 0.)[1:-1]
    vm = np.fmin(state.c_centers[-2], 0.)[1:-1]
    # v*grad_f
    return vp*diff_x_up(state.f) + vm*diff_x_down(state.f)


class BoltzmannSolver(object):

    def __init__(self, kn, riemann_solver, collision_operator, stepper='proj_int'):
        self.convection = riemann_solver
        self.collision = collision_operator
        self.kn = kn
        # step
        self.set_time_stepper(stepper)

    def dfdt(self, state):
        dx = state.delta[0]
        return -self.convection(state)/dx + 1/self.kn*self.collision(state.f)[1:-1]

    def euler(self, solution, Dt, dt, nt):
        solution.state.f[1:-1] += Dt*self.dfdt(solution.state)

    def proj_int(self, solution, Dt, dt, nt):
        state = solution.state
        for _ in tnrange(nt, desc='Inner', leave=False):
            state.f[1:-1] += dt*self.dfdt(state)
        state.f[1:-1] += (Dt - nt*dt)*self.dfdt(state)

    def set_time_stepper(self, stepper='euler'):
        """Sets the time step scheme to be used while solving given a
        string which should be one of ['euler', 'proj_int']."""
        if stepper == 'euler':
            self.step = self.euler
        elif stepper == 'proj_int':
            self.step = self.proj_int

    def solve(self, solution, Dt, Nt, dt, nt):
        for _ in tnrange(Nt, desc='Outer'):
            self.step(solution, Dt, dt, nt)
