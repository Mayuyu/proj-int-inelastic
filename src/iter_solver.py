import logging
import math

import numpy as np

from .utility import set_logger


class IterativeBoltzmannSolver(object):
    def __init__(self, log_path, collision=None, kn=0.1, tol=1e-6):
        set_logger(log_path)

        self.collision = collision
        self.kn = kn
        self.tol = tol

    def solve(self, solution, n_iter=0, disp=False):
        err = 1
        num_steps = 1
        while err > self.tol:
            if n_iter and num_steps >= n_iter:
                return err, num_steps
            err = self.step(solution.state)
            num_steps += 1
            if disp and (num_steps % 1 == 0):
                logging.info(
                    "Current error: {}; Number of iterations: {}".format(
                        err, num_steps
                    )
                )

        return err, num_steps

    def step(self, solution):
        raise NotImplementedError("No stepping routine has been defined!")


class BoltzmannFastSweep(IterativeBoltzmannSolver):
    def step(self, state):
        """Iterate one step."""
        f_old = state.f.copy()

        nx, dx = state.num_cells[0], state.delta[0]
        v = state.problem_data["v1"][:, 0]
        vdx = v / dx
        nu = 1 / np.sqrt(math.pi / 2) / self.kn

        rho = state.rho[:, None, None]
        gain = self.collision(state.f)
        for i in range(1, nx - 1):
            state.f[i, v > 0] = (
                nu * gain[i, v > 0]
                + vdx[v > 0][:, None] * state.f[i - 1, v > 0]
            ) / (nu * rho[i] + (vdx[v > 0])[:, None])
        for j in range(1, nx - 1):
            state.f[-j - 1, v < 0] = (
                nu * gain[-j - 1, v < 0]
                - vdx[v < 0][:, None] * state.f[-j, v < 0]
            ) / (nu * rho[-j - 1] - (vdx[v < 0])[:, None])

        # return np.sqrt(np.sum((state.rho - rho_old) ** 2) * state.grid.dx)
        return np.max(np.abs(state.f - f_old))
