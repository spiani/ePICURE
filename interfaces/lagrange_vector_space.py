import numpy as np
from scipy.interpolate import lagrange
from interfaces.vector_space import MonodimensionalVectorSpace

class LagrangeVectorSpace(MonodimensionalVectorSpace):
    """A python interface used to describe *one dimensional lagrange basis functions
    functions* on the interval given by the first and last interpolation point. 
    """
    def __init__(self, qpoints, continuous=True):
        """Generates the lagrange basis functions based on the given quadrature points.
        """
        n_dofs = len(qpoints)
        if continuous:
            assert n_dofs > 1, \
              'We can only have continuous functions if we have at least 2 n_dofs!'
            n_dofs_per_end_point = 1
        else:
            n_dofs_per_end_point = 0
        cells = np.array([qpoints[0], qpoints[-1]])
        super(LagrangeVectorSpace, self).__init__(cells, n_dofs,
						  n_dofs_per_end_point)
        self.qpoints = qpoints
        self.degree = n_dofs-1

    def basis(self, i):
        self.check_index(i)
        w = self.qpoints*0
        w[i] = 1.0
        return lagrange(self.qpoints, w)

    def basis_der(self, i, d):
        self.check_index(i)
        return self.basis(i).deriv(d)

    def cell_span(self, i):
        assert i==0, 'We only have one cell!'
        return range(self.n_dofs)
      
    def basis_span(self, i):
        self.check_index(i)
        return (0, 1)

class UniformLagrangeVectorSpace(LagrangeVectorSpace):
    """A python interface used to describe *one dimensional lagrange basis functions
    functions* on n equispaced quadrature points. 
    """
    def __init__(self, n, continuous=True):
         LagrangeVectorSpace.__init__(self, np.linspace(0, 1, n), continuous)
