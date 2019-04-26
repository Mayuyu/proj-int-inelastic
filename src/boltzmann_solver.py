import numpy as np
from tqdm import tnrange

from .limiters import tvd
from .solver import Solver


class BoltzmannSolver(Solver):

    def __init__(self, kn=1., riemann_solver=None, collision_operator=None):

        self.kn = kn
        self.num_ghost = 2
        self.order = 2
        self.limiters = tvd.minmod

        self.time_integrator = 'Euler'
        # Used only if time integrator is 'PFE'or 'TPI3'
        self.num_levels = 3
        self.inner_steps = [3, 3, 3]
        self.inner_dt = [kn, 5*kn, 10*kn]

        # Call general initialization function
        super(BoltzmannSolver, self).__init__(
            riemann_solver, collision_operator)

    def setup(self, solution):
        r"""
        Perform essential solver setup.
        """
        self._allocate_bc_arrays(solution.state)
        super(BoltzmannSolver, self).setup(solution)

    def step(self, solution, take_one_step, tstart, tend):
        r"""
        Evolve solution one time step.
        """
        if self.time_integrator == 'Euler':
            self.euler_step(solution, take_one_step, tstart, tend)
        elif self.time_integrator == 'PFE':
            self.proj_euler_step(solution, take_one_step, tstart, tend)
        elif self.time_integrator == 'TPI2':
            self.tel_proj_step2(solution, take_one_step, tstart, tend)
        elif self.time_integrator == 'TPI3':
            self.tel_proj_step3(solution, take_one_step, tstart, tend)
        else:
            raise NotImplementedError("This time integrator is not implemented.")

    # =============== Time stepping routines ======================
    def euler_step(self, solution, take_one_step, tstart, tend):
        state = solution.state
        state.f += self.df(state, self.dt)

    def proj_euler_step(self, solution, take_one_step, tstart, tend):
        state = solution.state
        M = self.dt/self.inner_dt[0] - self.inner_steps[0]

        for _ in tnrange(self.inner_steps[0], desc='Inner', leave=False):
            df = self.df(state, self.inner_dt[0])
            state.f += df

        state.f += M * df

    def tel_proj_step2(self, solution, take_one_step, tstart, tend):
        state = solution.state
        M1 = self.dt/self.inner_dt[1] - self.inner_steps[1]
        for _ in tnrange(self.inner_steps[1], desc='Level 1', leave=False):
            M0 = self.inner_dt[1]/self.inner_dt[0] - self.inner_steps[0]
            for _ in tnrange(self.inner_steps[0], desc='Level 0', leave=False):
                df0 = self.df(state, self.inner_dt[0])
                state.f += df0
            df1 = M0 * df0
            state.f += df1
        state.f += M1*df1

    def tel_proj_step3(self, solution, take_one_step, tstart, tend):
        state = solution.state
        M2 = self.dt/self.inner_dt[2] - self.inner_steps[2]
        for _ in tnrange(self.inner_steps[2], desc='Level 2', leave=False):
            M1 = self.inner_dt[2]/self.inner_dt[1] - self.inner_steps[1]
            for _ in tnrange(self.inner_steps[1], desc='Level 1', leave=False):
                M0 = self.inner_dt[1]/self.inner_dt[0] - self.inner_steps[0]
                for _ in tnrange(self. inner_steps[0], desc='Level 0', leave=False):
                    df0 = self.df(state, self.inner_dt[0])
                    state.f += df0
                df1 = M0*df0
                state.f += df1
            df2 = M1*df1
            state.f += df2
        state.f += M2*df2

    def df(self, state, dt):
        deltaf = self.df_advection(state, dt)
        if self.collision is not None:
            deltaf += dt * self.collision(state.f) / self.kn
        return deltaf

    def df_advection(self, state, dt):
        raise NotImplementedError('You must subclass the BoltzmannSolver.')


# ==================== Boltzmann solver for 1D in x ========================
class BoltzmannSolver1D(BoltzmannSolver):

    def __init__(self, kn=1, riemann_solver=None, collision_operator=None):
        self.num_dim_x = 1
        super(BoltzmannSolver1D, self).__init__(
            kn, riemann_solver, collision_operator)

    def df_advection(self, state, dt):
        # Apply boundary condition
        self._apply_bcs(state)
        f = self.fbc
        grid = state.grid
        dtdx = dt / grid.delta[0]
        df = np.zeros(f.shape)

        # limiter = np.array(self._mthlim, ndimn=1)
        # Solve Riemann problem at each interface
        f_l = f[:-1]
        f_r = f[1:]
        wave, s, amdf, apdf = self.rp(f_l, f_r, state.problem_data)

        # Loop limits for local portion of grid
        LL = self.num_ghost - 1
        UL = grid.num_cells[0] + self.num_ghost + 1
        df[LL:UL] = -dtdx * (amdf[LL:UL]+apdf[LL-1:UL-1])

        if self.order == 2:
            # Initialize flux corrections
            F = np.zeros(f.shape)
            wave = tvd.limit(wave, s, self.limiters, dtdx)
            sabs = np.abs(s[LL-1:UL-1])
            om = 1.0 - sabs*dtdx
            F[LL:UL] = 0.5 * sabs * om * wave[LL-1:UL-1]
            df[LL:UL-1] -= dtdx * (F[LL+1:UL]-F[LL:UL-1])

        return df[self.num_ghost:-self.num_ghost]

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