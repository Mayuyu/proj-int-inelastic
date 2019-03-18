import numpy as np


class Domain(object):

    @property
    def num_dim(self):
        return self._get_base_grid_attribute('num_dim')

    @property
    def grid(self):
        return self.grids[0]

    def __init__(self, *args):
        if len(args) > 1:
            lower = args[0]
            upper = args[1]
            n = args[2]
            dims = []
            names = ['x', 'vx', 'vy', 'vz']
            names = names[:len(n) + 1]
            for low, up, nn, name in zip(lower, upper, n, names):
                dims.append(Dimension(low, up, nn, name=name))
            self.grids = [Grid(dims)]
        else:
            geom = args[0]
            if not isinstance(geom, list) and not isinstance(geom, tuple):
                geom = [geom]
            if isinstance(geom[0], Grid):
                self.grids = geom
            elif isinstance(geom[0], Dimension):
                self.grids = [Grid(geom)]

    def _get_base_grid_attribute(self, name):
        return getattr(self.grids[0], name)

    def __deepcopy__(self, memo={}):
        import copy
        result = self.__class__(copy.deepcopy(self.grids))
        result.__init__(copy.deepcopy(self.grids))

        return result


class Grid(object):

    def __getattr__(self, key):
        if key in ['num_cells', 'lower', 'upper', 'delta', 'centers', 'nodes']:
            return self.get_dim_attribute(key)
        else:
            raise ArithmeticError("'Grid' object has no attribute '"+key+"'")

    @property
    def num_dim(self):
        return len(self._dimensions)

    @property
    def num_dim_x(self):
        return self._num_dim_x

    @num_dim_x.setter
    def num_dim_x(self, num_dim_x):
        self._num_dim_x = num_dim_x

    @property
    def dimensions(self):
        return [getattr(self, name) for name in self._dimensions]

    @property
    def c_centers(self):
        self._compute_c_centers()
        return self._c_centers

    @property
    def c_nodes(self):
        self._compute_c_nodes()
        return self._c_nodes

    def __init__(self, dimensions):

        self._c_centers = None
        self._c_nodes = None
        self._num_dim_x = None

        if isinstance(dimensions, Dimension):
            dimensions = [dimensions]
        self._dimensions = []
        for dim in dimensions:
            self.add_dimension(dim)

        super(Grid, self).__init__()

    def _clear_cached_values(self):
        self._c_centers = None
        self._c_nodes = None

    def add_dimension(self, dimension):
        if dimension.name in self._dimensions:
            raise Exception('Unable to add dimensions. A dimension' +
                            ' of the same name: {name}, already exists.'.format(name=dimension.name))

        self._dimensions.append(dimension.name)
        setattr(self, dimension.name, dimension)
        self._clear_cached_values()

    def get_dim_attribute(self, attr):
        return [getattr(dim, attr) for dim in self.dimensions]

    def __copy__(self):
        return self.__class__(self)

    def __str__(self):
        output = "%s-dimensional domain " % str(self.num_dim)
        output += "("+",".join([dim.name for dim in self.dimensions])+")\n"
        output += " x ".join(["[{:.2}, {:.2}]".format(dim.lower, dim.upper)
                              for dim in self.dimensions])
        output += "\n"
        output += "Cells:  "
        output += " x ".join(["{}".format(dim.num_cells)
                              for dim in self.dimensions])
        return output

    def _compute_c_centers(self, recompute=False):
        if recompute or (self._c_centers is None) or any([c is None for c in self.get_dim_attribute('_centers')]):
            index = np.indices(self.num_cells)
            self._c_centers = []
            for i, center_array in enumerate(self.get_dim_attribute('centers')):
                self._c_centers.append(center_array[index[i, ...]])

    def _compute_c_nodes(self, recompute=False):
        if recompute or (self._c_nodes is None) or any([c is None for c in self.get_dim_attribute('_nodes')]):
            index = np.indices(n+1 for n in self.num_cells)
            self._c_nodes = []
            for i, nodes_array in enumerate(self.get_dim_attribute('nodes')):
                self._c_nodes.append(nodes_array[index[i, ...]])

    def c_center(self, ind):
        index = [np.array(i) for i in ind]
        return np.array([self.c_centers[i][index] for i in range(self.num_dim)])

    def __deepcopy(self, memo={}):
        import copy
        result = self.__class__(copy.deepcopy(self.dimensions))
        result.__init__(copy.deepcopy(self.dimensions))

        for attr in ('_c_centers', '_c_nodes', '_num_dim_x'):
            setattr(result, attr, getattr(self, attr))

        return result


class Dimension(object):

    @property
    def delta(self):
        return (self.upper - self.lower) / float(self.num_cells)

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = np.empty(self.num_cells+1)
            for i in range(self.num_cells+1):
                self._nodes[i] = self.lower + i*self.delta
        return self._nodes

    @property
    def centers(self):
        if self._centers is None:
            self._centers = np.empty(self.num_cells)
            for i in range(self.num_cells):
                self._centers[i] = self.lower + (i+0.5)*self.delta
        return self._centers

    @property
    def lower(self):
        return self._lower

    @lower.setter
    def lower(self, lower):
        self._lower = float(lower)
        self._centers = None
        self._nodes = None
        self._check_validity()

    @property
    def upper(self):
        return self._upper

    @upper.setter
    def upper(self, upper):
        self._upper = float(upper)
        self._centers = None
        self._nodes = None
        self._check_validity()

    @property
    def num_cells(self):
        return self._num_cells

    @num_cells.setter
    def num_cells(self, num_cells):
        self._num_cells = int(num_cells)
        self._centers = None
        self._nodes = None
        self._check_validity()

    def centers_with_ghost(self, num_ghost):
        r"""(ndarrary(:)) - Location of all cell center coordinates
        for this dimension, including centers of ghost cells."""
        centers = self.centers
        pre = self.lower+(np.arange(-num_ghost, 0)+0.5)*self.delta
        post = self.upper + self.delta * (np.arange(num_ghost) + 0.5)
        return np.hstack((pre, centers, post))

    def nodes_with_ghost(self, num_ghost):
        r"""(ndarrary(:)) - Location of all edge coordinates
        for this dimension, including nodes of ghost cells."""
        nodes = self.nodes
        pre = np.linspace(self.lower-num_ghost*self.delta,
                          self.lower-self.delta, num_ghost)
        post = np.linspace(self.upper+self.delta, self.upper +
                           num_ghost*self.delta, num_ghost)
        return np.hstack((pre, nodes, post))

    def __init__(self, lower, upper, num_cells, name='x'):

        self._nodes = None
        self._centers = None

        self._lower = float(lower)
        self._upper = float(upper)
        self._num_cells = int(num_cells)
        self.name = name

        self._check_validity()

    def _check_validity(self):
        assert isinstance(
            self.num_cells, int), 'Dimension.num_cells must be an integer; got %s' % type(self.num_cells)
        assert isinstance(self.lower, float), 'Dimension.lower must be a float'
        assert isinstance(self.upper, float), 'Dimension.upper must be a float'
        assert self.num_cells > 0, 'Dimension.num_cells must be positive'
        assert self.upper > self.lower, 'Dimension.upper must be greater than lower'

    def __str__(self):
        output = "Demension %s" % self.name
        output += ": (num_cells,delta,[lower,upper]) = (%s,%s,[%s,%s])" % (
            self.num_cells, self.delta, self.lower, self.upper)
        return output

    def __len__(self):
        return self.num_cells
