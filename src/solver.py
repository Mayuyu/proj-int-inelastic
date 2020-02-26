import numpy as np
from tqdm import tnrange


class BC:
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

        self.bc_lower = [None] * (self.num_dim_x)
        self.bc_upper = [None] * (self.num_dim_x)

        self._is_set_up = False

        super(Solver, self).__init__()

    def setup(self, solution):

        self._is_set_up = True

    def _allocate_bc_arrays(self, state):
        fbc_dim = [
            n + 2 * self.num_ghost
            for n in state.grid.num_cells[: self.num_dim_x]
        ] + [n for n in state.grid.num_cells[self.num_dim_x :]]
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
            array[: self.num_ghost, ...] = array[
                -2 * self.num_ghost : -int(self.num_ghost), ...
            ]
        else:
            if bc_type is None:
                raise Exception("Lower boundary condition not specified.")
            else:
                raise NotImplementedError(
                    "Boundary condition %s not implemented" % bc_type
                )

    def _bc_upper(self, bc_type, array):
        r"""
        Apply upper boundary conditions to array.
        """

        if bc_type == BC.extrap:
            for i in range(self.num_ghost):
                array[-i - 1, ...] = array[-int(self.num_ghost) - 1, ...]
        elif bc_type == BC.periodic:
            array[-int(self.num_ghost) :, ...] = array[
                self.num_ghost : 2 * self.num_ghost, ...
            ]
        else:
            if bc_type is None:
                raise Exception("Upper boundary condition not specified.")
            else:
                raise NotImplementedError(
                    "Boundary condition %s not implemented" % bc_type
                )

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
                if abs(self.max_steps * self.dt - (tend - tstart)) > 1e-5 * (
                    tend - tstart
                ):
                    raise Exception(
                        "dt does not divide (tend-tstart) and dt is fixed!"
                    )

        # Main time-stepping loop
        for n in tnrange(self.max_steps):

            if self.before_step is not None:
                self.before_step(self, solution.state)

            self.step(solution, take_one_step, tstart, tend)

            solution.t = tstart + (n + 1) * self.dt

            # if take_one_step:
            #     break
            # elif solution.t >= tend:
            #     break

    def step(self, solution, take_one_step, tstart, tend):
        raise NotImplementedError("No stepping routine has been defined!")
