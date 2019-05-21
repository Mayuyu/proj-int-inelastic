# encoding: utf-8
r"""
Euler 2D Quadrants example
==========================

Simple example solving the Euler equations of compressible fluid dynamics:

.. math::
    \rho_t + (\rho u)_x + (\rho v)_y & = 0 \\
    (\rho u)_t + (\rho u^2 + p)_x + (\rho uv)_y & = 0 \\
    (\rho v)_t + (\rho uv)_x + (\rho v^2 + p)_y & = 0 \\
    E_t + (u (E + p) )_x + (v (E + p))_y & = 0.

Here :math:`\rho` is the density, (u,v) is the velocity, and E is the total energy.
The initial condition is one of the 2D Riemann problems from the paper of
Liska and Wendroff.

"""
from __future__ import absolute_import
from clawpack import pyclaw
from clawpack import riemann
from clawpack.riemann.euler_4wave_2D_constants import density, x_momentum, y_momentum, \
        energy, num_eqn
from clawpack.visclaw import colormaps

def q_src(solver, state, dt):
    q = state.q
    q[3] += 0.5*dt*(0.5*(q[1]**2 + q[2]**2) - q[0]*q[3])


class Euler2D(object):

    def __init__(self):

        self.setup()

    def setup(self):
        solver = pyclaw.ClawSolver2D(riemann.euler_4wave_2D)
        solver.all_bcs = pyclaw.BC.periodic

        domain = pyclaw.Domain([0.,0.],[1.,1.],[100,100])
        solution = pyclaw.Solution(num_eqn,domain)
        gamma = 2.0
        solution.problem_data['gamma']  = gamma
        solver.dimensional_split = False
        solver.transverse_waves = 2
        solver.step_source = q_src
        
        # Set initial data
        xx, yy = domain.grid.p_centers
        l = xx < 0.8
        r = xx >= 0.8
        b = yy < 0.8
        t = yy >= 0.8
        solution.q[density,...] = 1.5 * r * t + 0.532258064516129 * l * t          \
                                            + 0.137992831541219 * l * b          \
                                            + 0.532258064516129 * r * b
        u = 0.0 * r * t + 1.206045378311055 * l * t                                \
                        + 1.206045378311055 * l * b                                \
                        + 0.0 * r * b
        v = 0.0 * r * t + 0.0 * l * t                                              \
                        + 1.206045378311055 * l * b                                \
                        + 1.206045378311055 * r * b
        p = 1.5 * r * t + 0.3 * l * t + 0.029032258064516 * l * b + 0.3 * r * b
        solution.q[x_momentum,...] = solution.q[density, ...] * u
        solution.q[y_momentum,...] = solution.q[density, ...] * v
        solution.q[energy,...] = 0.5 * solution.q[density,...]*(u**2 + v**2) + p / (gamma - 1.0)

        claw = pyclaw.Controller()
        claw.tfinal = 0.8
        claw.solution = solution
        claw.solver = solver

        claw.output_format = 'ascii'    
        claw.outdir = "./_output"

        self._claw = claw

    def run(self):
        self._claw.run()
        self.plot()

    def plot(self):
        import matplotlib.pyplot as plt
        _, axes = plt.subplots(figsize=(5, 5))
        axes.contourf(self._claw.solution.q[0])
        plt.show()


if __name__ == "__main__":
    euler = Euler2D()
    euler.run()

