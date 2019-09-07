import numpy as np
from clawpack import pyclaw, riemann
from clawpack.riemann.euler_with_efix_1D_constants import (density, energy,
                                                           momentum, num_eqn)

# Ratio of specific heats
gamma = 2.0

def q_src(solver, state, dt):
    state.q[2] += 0.5*dt*(0.5*state.q[1]**2 - state.q[0]*state.q[2]) + 2*state.q[0]*dt
    
def dq_src(solver, state, dt):
    dq = np.zeros(state.q.shape)
    dq[2] = 0.5*dt*(0.5*state.q[1]**2 - state.q[0]*state.q[2]) + 2*state.q[0]*dt            
    return dq
        
class Euler1D(object):

    def __init__(self, domain_config, solver_type='sharpclaw', kernel_language='Python'):
        if kernel_language =='Python':
            rs = riemann.euler_1D_py.euler_hllc_1D
        elif kernel_language =='Fortran':
            rs = riemann.euler_with_efix_1D

        if solver_type =='sharpclaw':
            solver = pyclaw.SharpClawSolver1D(rs)
            # solver.dq_src = dq_src
        elif solver_type =='classic':
            solver = pyclaw.ClawSolver1D(rs)
            # solver.step_source = q_src
        solver.kernel_language = kernel_language

        solver.bc_lower[0]=pyclaw.BC.extrap
        solver.bc_upper[0]=pyclaw.BC.extrap

        nx = domain_config.nx
        xmin, xmax = domain_config.xmin, domain_config.xmax
        x = pyclaw.Dimension(xmin, xmax, nx, name='x')
        domain = pyclaw.Domain([x])

        state = pyclaw.State(domain, num_eqn)
        state.problem_data['gamma'] = gamma
        state.problem_data['gamma1'] = gamma - 1.

        solution = pyclaw.Solution(state, domain)

        claw = pyclaw.Controller()
        claw.solution = solution
        claw.solver = solver

        self._solver = solver
        self._domain = domain
        self._solution = solution
        self._claw = claw

    def set_initial(self, rho, u, T):
        soln = self._solution
        soln.q[density ,:] = rho
        soln.q[momentum,:] = rho * u
        soln.q[energy  ,:] = 0.5*rho*u**2 + rho*T

    def solve(self, tfinal):
        self._claw.tfinal = tfinal
        self._claw.run() 

    def plot(self):
        pass

    @property
    def macros(self):
        soln = self._solution
        rho = soln.q[density, :]
        u = soln.q[momentum,:] / rho
        T = soln.q[energy , :] / rho - 0.5*u**2
        return rho, u, T