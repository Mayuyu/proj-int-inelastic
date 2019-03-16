from math import pi

import matplotlib.pyplot as plt
import numpy as np
from limiters import tvd
from tqdm import tnrange


class BC():
    extrap = 1
    periodic = 2


class Solver(object):

    @property
    def all_bcs(self):
        return self.bc_lower, self.bc_upper

    @all_bcs.setter
    def all_bcs(self, all_bcs):
        for i in range(self.num_dim_x):
            self.bc_lower[i] = all_bcs
            self.bc_upper[i] = all_bcs

    def __init__(self, riemann_solver=None, collision_operator=None):
        r"""
        Initialize a Solver object
        """

        self.dt_initial = 0.01
        self.dt_max = 1e99
        self.max_steps = 10000
        self.dt_variable = False

        self.fbc = None

        if riemann_solver is not None:
            self.rp = riemann_solver
        self.collision = collision_operator

        self.before_step = None

        self.dt = self.dt_initial

        self.bc_lower = [None]*(self.num_dim_x)
        self.bc_upper = [None]*(self.num_dim_x)

        self._is_set_up = False

        super(Solver, self).__init__()

    def setup(self, solution):

        self._is_set_up = True

    def _allocate_bc_arrays(self, state):
        fbc_dim = [
            n+2*self.num_ghost for n in state.grid.num_cells[:self.num_dim_x]] + [n for n in state.grid.num_cells[self.num_dim_x:]]
        self.fbc = np.zeros(fbc_dim)
        self._apply_bcs(state)

    def _apply_bcs(self, state):
        self.fbc = state.get_fbc_from_f(self.num_ghost, self.fbc)
        grid = state.grid

        for idim in range(grid.num_dim_x):
            self._bc_lower(self.bc_lower[idim], np.rollaxis(self.fbc, idim))
            self._bc_upper(self.bc_upper[idim], np.rollaxis(self.fbc, idim))

    def _bc_lower(self, bc_type, array):
        r"""
        Apply lower boundary conditions to array.
        """

        if bc_type == BC.extrap:
            for i in range(self.num_ghost):
                array[i, ...] = array[self.num_ghost, ...]
        elif bc_type == BC.periodic:
            array[:self.num_ghost, ...] = array[-2 *
                                                self.num_ghost:-self.num_ghost, ...]
        else:
            if bc_type is None:
                raise Exception('Lower boundary condition not specified.')
            else:
                raise NotImplementedError(
                    "Boundary condition %s not implemented" % bc_type)

    def _bc_upper(self, bc_type, array):
        r"""
        Apply upper boundary conditions to array.
        """

        if bc_type == BC.extrap:
            for i in range(self.num_ghost):
                array[-i-1, ...] = array[-self.num_ghost-1, ...]
        elif bc_type == BC.periodic:
            array[-self.num_ghost:, ...] = array[self.num_ghost:2*self.num_ghost, ...]
        else:
            if bc_type is None:
                raise Exception('Upper boundary condition not specified.')
            else:
                raise NotImplementedError(
                    "Boundary condition %s not implemented" % bc_type)

    def evolve_to_time(self, solution, tend=None):
        if not self._is_set_up:
            self.setup(solution)

        if tend is None:
            take_one_step = True
        else:
            take_one_step = False

        tstart = solution.t
    
        if not self.dt_variable:
            if take_one_step:
                self.max_steps = 1
            else:
                self.max_steps = int((tend - tstart + 1e-10) / self.dt)
                if abs(self.max_steps*self.dt - (tend - tstart)) >      \
                   1e-5 * (tend - tstart):
                    raise Exception(
                        'dt does not divide (tend-tstart) and dt is fixed!')

        # Main time-stepping loop
        for n in tnrange(self.max_steps):

            if self.before_step is not None:
                self.before_step(self, solution.state)

            self.step(solution, take_one_step, tstart, tend)

            solution.t = tstart + (n+1)*self.dt

            # if take_one_step:
            #     break
            # elif solution.t >= tend:
            #     break

    def step(self, solution):
        raise NotImplementedError("No stepping routine has been defined!")


class BoltzmannSolver(Solver):

    def __init__(self, kn=1., riemann_solver=None, collision_operator=None):

        self.kn = kn
        self.num_ghost = 2
        self.order = 2
        self.limiters = tvd.minmod
        self.time_integrator = 'Euler'

        # Call general initialization function
        super(BoltzmannSolver, self).__init__(riemann_solver, collision_operator)

    def setup(self, solution):
        r"""
        Perform essential solver setup.
        """
        self._allocate_bc_arrays(solution.state)

        super(BoltzmannSolver, self).setup(solution)

    # =============== Time stepping routines ======================

    def step(self, solution, take_one_step, tstart, tend):
        r"""
        Evolve solution one time step.
        """
        state = solution.state
        self.df_dt = self.df(state) / self.dt

        if self.time_integrator == 'Euler':
            state.f += self.dt*self.df_dt

    def df(self, state):
        deltaf = self.df_advection(state)
        if self.collision is not None:
            deltaf += self.dt*self.collision(state.f) / self.kn

        return deltaf
            
    def df_advection(self, state):
        raise NotImplementedError('You must subclass BoltzmannSolver.')

    # def rk4(self, solution, dt):
    #     state = solution.state
    #     f0 = state.f.copy()[1:-1]
    #     k1 = self.dfdt(state)
    #     state.f[1:-1] = f0 + 0.5*k1
    #     k2 = self.dfdt(state)
    #     state.f[1:-1] = f0 + 0.5*k2
    #     k3 = self.dfdt(state)
    #     state.f[1:-1] = f0 + 0.5*k3
    #     k4 = self.dfdt(state)

    #     state.f[1:-1] = f0 + dt*(1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4)

    # def pfe(self, solution, Dt, dt, nt):
    #     state = solution.state

    #     for _ in tnrange(nt, desc='Inner', leave=False):
    #         self.euler(solution, dt)
    #     state.f[1:-1] = state.f[1:-1] + (Dt - nt*dt)*self.dfdt(state)

    # def prk4(self, solution, Dt, dt, nt):
    #     state = solution.state

    #     for _ in tnrange(nt, desc='k1', leave=False):
    #         self.euler(solution, dt)
    #     k1 = self.dfdt(solution.state)
    #     f_nt_1 = (dt*k1 + state.f[1:-1]).copy()

    #     state.f[1:-1] = f_nt_1 + (0.5*Dt - (nt+1)*dt)*k1
    #     for _ in tnrange(nt, desc='k2', leave=False):
    #         self.euler(solution, dt)
    #     k2 = self.dfdt(state)

    #     state.f[1:-1] = f_nt_1 + (0.5*Dt - (nt+1)*dt)*k2
    #     for _ in tnrange(nt, desc='k3', leave=False):
    #         self.euler(solution, dt)
    #     k3 = self.dfdt(state)

    #     state.f[1:-1] = f_nt_1 + (Dt - (nt+1)*dt)*k3
    #     for _ in tnrange(nt, desc='k4', leave=False):
    #         self.euler(solution, dt)
    #     k4 = self.dfdt(state)

    #     state.f[1:-1] = f_nt_1 + (Dt - (nt+1)*dt) * \
    #         (1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4)


class BoltzmannSolver1D(BoltzmannSolver):

    def __init__(self, kn=1, riemann_solver=None, collision_operator=None):
        self.num_dim_x = 1

        super(BoltzmannSolver1D, self).__init__(kn, riemann_solver, collision_operator)

    def df_advection(self, state):
        # Apply boundary condition
        self._apply_bcs(state)
        f = self.fbc
        
        grid = state.grid
        dtdx = self.dt / grid.delta[0]
        df = np.zeros(f.shape)

        # limiter = np.array(self._mthlim, ndimn=1)
        # Solve Riemann problem at each interface
        f_l = f[:-1]
        f_r = f[1:]
        wave, s, amdf, apdf = self.rp(f_l, f_r, state.problem_data)

        # Loop limits for local portion of grid
        LL = self.num_ghost - 1
        UL = grid.num_cells[0] + self.num_ghost + 1

        df[LL:UL] = -dtdx*(amdf[LL:UL] + apdf[LL-1:UL-1])

        if self.order == 2:
            # Initialize flux corrections
            F = np.zeros(f.shape)
            wave = tvd.limit(wave, s, self.limiters, dtdx)

            sabs = np.abs(s[LL-1:UL-1])
            om = 1.0 - sabs*dtdx
            F[LL:UL] = 0.5*sabs*om*wave[LL-1:UL-1]

            df[LL:UL-1] -= dtdx*(F[LL+1:UL] - F[LL:UL-1])

        return df[self.num_ghost:-self.num_ghost]


