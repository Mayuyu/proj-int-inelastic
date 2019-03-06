import numpy as np

from geometry import Dimension, Domain, Grid
from state import State


class Solution(object):

    def __getattr__(self, key):
        if key in ('t', 'f', 'num_dim', 'c_nodes', 'c_centers', 'num_cells', 'lower', 'upper', 'delta', 'centers', 'grid'):
            return self._get_base_state_attribute(key)
        else:
            raise AttributeError(
                "'Solution' object has no attribute '"+key+"'")

    def _get_base_state_attribute(self, name):
        return getattr(self.state, name)

    @property
    def grid(self):
        return self.domain.grid

    def __init__(self, config):
        # Set domain
        L = config.l
        self.domain = Domain((config.xmin, -L, -L), (config.xmax, L, L), (config.nx, config.nv, config.nv))
        # Set state
        self.state = State(self.domain)

    def __copy__(self):
        return self.__class__(self)
