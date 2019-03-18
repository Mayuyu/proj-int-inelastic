import numpy as np

from src.geometry import Domain, Grid


class State(object):
    """A grid class that stores the details and solution of the
    computational grid."""

    def __getattr__(self, key):
        if key in ('num_dim', 'num_dim_x', 'c_centers', 'c_nodes', 'num_cells', 'lower', 'upper', 'delta', 'centers'):
            return self._get_grid_attribute(key)
        else:
            raise AttributeError("'State' object has no attribute '"+key+"'")

    def _get_grid_attribute(self, name):
        return getattr(self.grid, name)

    @property
    def rho(self):
        """Compute the macro quantity: density."""
        return self.sum_f(self.f)

    @property
    def u(self):
        return [self.sum_f(self.f*self.c_centers[i]) for i in (-2, -1)]

    @property
    def T(self):
        E = 0.5*self.sum_f(self.f*(self.c_centers[-2]**2 + self.c_centers[-1]**2))
        u_square = 0.
        for u in self.u:
            u_square += u**2

        return E / self.rho - 0.5*u_square


    def __init__(self, geom):
        if isinstance(geom, Grid):
            self.grid = geom
        elif isinstance(geom, Domain):
            self.grid = geom.grids[0]
        else:
            raise Exception("Must be initialzed with a Grid object.")

        self.problem_data = {}
        self.t = 0.
        self.f = self.new_array()

    def new_array(self):
        shape = []
        shape.extend(self.grid.num_cells)
        return np.empty(shape)

    def __copy__(self):
        return self.__class__(self)

    def __deepcopy__(self, memo={}):
        import copy
        result = self.__class__(copy.deepcopy(self.grid))
        result.__init__(copy.deepcopy(self.grid))

        for attr in ('t', 'problem_data'):
            setattr(result, attr, copy.deepcopy(getattr(self, attr)))

        if self.f is not None:
            result.f = copy.deepcopy(self.f)
        
        return result

    def get_f_global(self):
        return self.f.copy()

    def sum_f(self, f):
        return np.sum(f, axis=(-1, -2))*self.delta[-1]*self.delta[-2]

    def set_f_from_fbc(self, num_ghost, fbc):
        """
        Set the value of q using the array fbc. Typically this is called
        after fbc has been updated by the solver.
        """

        num_dim = self.grid.num_dim_x

        if num_dim == 1:
            self.f = fbc[num_ghost:-num_ghost]
        elif num_dim == 2:
            self.f = fbc[num_ghost:-num_ghost, num_ghost:-num_ghost]
        else:
            raise Exception("Assumption (1 <= num_dim_x <= 2) violated.")

    def get_fbc_from_f(self, num_ghost, fbc):
        """
        Fills in the interior of fbc by copying q to it.
        """
        num_dim = self.grid.num_dim_x

        if num_dim == 1:
            fbc[num_ghost:-num_ghost] = self.f
        elif num_dim == 2:
            fbc[num_ghost:-num_ghost, num_ghost:-num_ghost] = self.f
        else:
            raise Exception("Assumption (1 <= num_dim <= 2) violated.")

        return fbc
