import numpy as np

from geometry import Domain, Grid


class State(object):
    """A grid class that stores the details and solution of the
    computational grid."""

    def __getattr__(self, key):
        if key in ('num_dim', 'c_centers', 'c_nodes', 'num_cells', 'lower', 'upper', 'delta', 'centers'):
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
    def E(self):
        return 0.5*self.sum(self.f*(self.c_centers[-2]**2 + self.c_centers[-1]**2))

    def __init__(self, geom):
        if isinstance(geom, Grid):
            self.grid = geom
        elif isinstance(geom, Domain):
            self.grid = geom.grids[0]
        else:
            raise Exception("Must be initialzed with a Grid object.")

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

        for attr in ('t'):
            setattr(result, attr, copy.deepcopy(getattr(self, attr)))

        if self.f is not None:
            result.f = copy.deepcopy(self.f)

        return result

    def get_f_global(self):
        return self.f.copy()

    def sum_f(self, f):
        return np.sum(f, axis=(-1, -2))*self.delta[-1]*self.delta[-2]

    # def plot_f(self, i, cl=1):
    #     """Plot the contour in velocity space for given index i"""
    #     dv = self.dv
    #     v = np.mgrid[self.vmin+dv/2:self.vmax+dv/2:dv,
    #                  self.vmin+dv/2:self.vmax+dv/2:dv]

    #     fig, ax = plt.subplots()
    #     cs = ax.contour(v[0], v[1], self.f[i])
    #     if cl == 1:
    #         ax.clabel(cs, inline=0.5)

    #     ax.grid(linestyle=':')
    #     plt.show()

    # def plot_macro(self, macro='velocity'):
    #     """Plot the macroscopic quantities: density, velocity and temperature."""
    #     fig, ax = plt.subplots()
    #     if macro == 'density':
    #         cs = ax.plot(self.x, self.rho)
    #         ax.set_ylabel(r'$\rho(x)$', fontsize='large')
    #     elif macro == 'temperature':
    #         cs = ax.plot(self.x, self.temperature())
    #         ax.set_ylabel(r'$T(x)$', fontsize='large')
    #     else:
    #         cs = ax.plot(self.x, self.velocity())
    #         ax.set_ylabel(r'$u(x)$', fontsize='large')

    #     ax.set_xlabel(r'$x$', fontsize='large')
    #     ax.grid(which='both', linestyle=':')
    #     plt.show()
