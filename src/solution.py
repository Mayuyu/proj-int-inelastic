import numpy as np

from geometry import Dimension, Domain, Grid
from state import State


class Solution(object):

    def __getattr__(self, key):
        if key in ('t', 'f', 'rho', 'u', 'T', 'num_dim', 'num_dim_x', 
                    'c_nodes', 'c_centers', 'num_cells', 'lower', 
                    'upper', 'delta', 'centers', 'grid'):
            return self._get_base_state_attribute(key)
        else:
            raise AttributeError(
                "'Solution' object has no attribute '"+key+"'")

    def __setattr__(self, key, value):
        if key in ('t'):
            self.set_all_states(key,value)
        else:
            self.__dict__[key] = value

    def set_all_states(self,attr,value,overwrite=True):
        if getattr(self.state, attr) is None or overwrite:
            setattr(self.state, attr,value) 

    def _get_base_state_attribute(self, name):
        return getattr(self.state, name)

    @property
    def grid(self):
        return self.domain.grid

    def __init__(self, config):
        # Set domain
        dim_x, dim_v = config.dim_x, config.dim_v
        lower = [config.xmin]*dim_x + [-config.l]*dim_v
        upper = [config.xmax]*dim_x + [config.l]*dim_v
        nn = [config.nx]*dim_x + [config.nv]*dim_v
        self.domain = Domain(lower, upper, nn)
        # Set state
        self.state = State(self.domain)
        self.state.grid.num_dim_x = dim_x
        self.state.problem_data = {'v1': self.c_centers[dim_x][0]}

    def __copy__(self):
        return self.__class__(self)
