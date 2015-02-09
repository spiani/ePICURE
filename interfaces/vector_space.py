import numpy as np
from collections import Iterable

from interfaces.cell import Rectangle
from interfaces.f_with_domain import FWithDomain, OutOfDomainError

class VectorSpace(object):
    """An abstract python interface used to describe some functions
    on a domain of R^n, as *coefficients* times *basis functions*,
    with access to single basis functions, their derivatives, their
    support and the splitting of the domain into "cells" where
    *each basis* is assumed to be smooth.

    This class provides an abstract interface to describe VectorSpaces
    of functions from R^k to  R^N = R^mxnxd...`.

    The main attribute of this class is the `element` method, which
    takes as input a numpy array of length VectorSpace.n_dofs

    len(C) == VectorSpace.n_dofs # True    
    g = VectorSpace.element(C)
    
    In general, this results in a callable function g, such that

    shape( g(x) ) = shape(C[0])+shape(x)
    
    g := C[i,j,k, ...]*VectorSpace.basis(i) # sum over i is implicit

    If you evaluate g at a nparray x, then
    
    C = g(x)
    C[j,k,...,l,m,...] := C[i,j,k, ...]*VectorSpace.basis(i)(x[l,m, ...])
    
    where again a sum is implicit in i.
    
    Optionally, you could assign a shape to the elements of the basis. For 
    example, the base could be e_ij. In that case, you must supply a vector
    c[i,j,...] and the sum will be performed on both the two indices
    """
    def __init__(self, n_dofs, base_shape = None,
                 domain_dimension=1):
        # base_shape is the shape, as a tuple, of the base. For example,
        # (10,10) means that there is a base of 100 elements with 2 index
        # to assign one
        self.n_dofs = n_dofs
        if base_shape is not None:
            self._base_shape = base_shape
        else:
            self._base_shape = (self.n_dofs,)
        self._domain_dimension = domain_dimension

    def _basis(index_tuple):
        # This is the most important function inside class that inherit
        # from VectorSpace and MUST be implemented. This function get
        # a tuple of index as input (like (1,3), for example) and
        # should return the function e_{1,3} as a callable object.
        # The output shall be an instance of FWithDomain. Please,
        # while reimplementing this function in your class, ensure
        # that the index_tuple is coherent with the value of
        # self._base_shape. This function should not be directly
        # called by a normal user; he/she can get the base elements
        # using element([1,0,0,0,...])
        raise NotImplementedError  
    
    def _basis_der(index_tuple, d):
        # For this function, everything that was written in the 
        # comment of _basis function still holds. The var d is a
        # tuple of indices that express the variables to be derivated
        # For example, d=(1,0,0) means derive respect to x, d=(0,1,0)
        # means derive respect to y
        raise NotImplementedError

    def eval(self, c, x, broadcast=0):
        ### THIS FUNCTION SHALL BE IMPROVED!!!
        # Now, it is too slow! We need to avoid recursion
        # from the second block to the first one!!!
       
       
        if c.shape == self._base_shape:
            def func_ev(i):
                indx = np.unravel_index(i,c.shape)
                coeff = c[indx]
                base_eval = self._basis(indx)(x, broadcast=broadcast)
                return coeff * base_eval
            # The original idea for this map was to implement some sort
            # of multithreading here, but maybe it was not a great idea
            all_eval = np.array(map(func_ev,xrange(c.size)))
            return np.sum(all_eval,axis = 0)
        
        # This is the case when there more indices in C than in the
        # base
        elif c.shape[:len(self._base_shape)] == self._base_shape:
            overindex = c.shape[len(self._base_shape):]
            eval_iterations = np.prod(overindex)
            def multi_eval(i):
                tmp_indx = np.unravel_index(i, overindex)
                indx = [Ellipsis]
                indx.extend(tmp_indx)
                indx = tuple(indx)
                return self.eval(c[indx], x, broadcast=broadcast)
            output = np.array(map(multi_eval, xrange(eval_iterations)))
            if isinstance(output[0],np.ndarray):
                overindex += output[0].shape
            return output.reshape(overindex)
        else:
            raise ValueError('The input variable c has a shape which is not'
                             ' compatible with the shape of the basis')
         
             
    def eval_der(self, c, d, x, broadcast=0):
        # See eval for some comments that still holds here
        if c.shape == self._base_shape:
            def func_ev(i):
                indx = np.unravel_index(i,c.shape)
                coeff = c[indx]
                try:
                    base_eval = self._basis_der(indx, d)(x, broadcast=broadcast)
                except OutOfDomainError:
                    return 0
                return coeff * base_eval
            all_eval = np.array(map(func_ev, xrange(c.size)))
            return np.sum(all_eval, axis = 0)
        elif c.shape[:len(self._base_shape)] == self._base_shape:
            overindex = c.shape[len(self._base_shape):]
            eval_iterations = np.prod(overindex)
            def multi_eval(i):
                tmp_indx = np.unravel_index(i, overindex)
                indx = [Ellipsis]
                indx.extend(tmp_indx)
                indx = tuple(indx)
                return self.eval_der(c[indx], d,  x, broadcast=broadcast)
            output = np.array(map(multi_eval, xrange(eval_iterations)))
            if isinstance(output[0],np.ndarray):
                overindex += output[0].shape
            return output.reshape(overindex)
        else:
            raise ValueError('The input variable c has a shape which is not'
                             ' compatible with the shape of the basis')

        
    def element(self, c, broadcast=0):
        """VectorSpace.element(c): a callable function,
        representing sum(c[i,j,...] * basis[i,j,...]) """
        return lambda x: self.eval(c,x, broadcast)

    def element_der(self, c, d, broadcast=0):
        """VectorSpace.element_der(c): a callable function,
        representing sum(c[i,j,...] * D(basis[i,j,...])).
        d shall be passed as a tuple. Every index is the number
        of times that derivation will be performed on the variabile
        of place index. For example (1,2,0) means "derive on time for
        x and two times for y"""
        return lambda x: self.eval_der(c, d, x, broadcast=broadcast)        
    



class MonodimensionalVectorSpace(VectorSpace):
    # This class is just a wrapper. It will maintain compatibility between
    # the VectorSpace classes written with the monodimensional version of
    # the VectorSpace class and the current one. It is not beatiful code,
    # but is a useful workaround!
    def __init__(self, cells, n_dofs, n_dofs_per_end_point = 0):
        super(MonodimensionalVectorSpace, self).__init__(n_dofs)
        self.n_cells = len(cells) - 1
        self.cells = cells
        self.n_dofs_per_end_point = n_dofs_per_end_point

    def check_index(self, i):
        assert i< self.n_dofs, \
            ('Trying to access base {}, but we only'
             'have {}'.format(i, self.n_dofs))

    def basis(self, i):
        # This is what should be reimplemented. This is expected to
        # return just a callable object that is the ith element of the
        # base of the vectorial space
        raise NotImplementedError

    def _basis(self,i):
        # This function is just a small trick. It gets the function
        # in the old form (as it is returned from basis), create a
        # cell object for that function and returns it in the form
        # of a FWithDomain istance
                
        # All we have to do is add a domain to the function
        assert i < self._base_shape, "Wrong base dimension!"
        func = self.basis(i[0])
        borders = self.basis_span(i[0])
        rect_cell = Rectangle(np.array((borders[0],)), 
                              np.array((borders[1],)))
        return FWithDomain(func, rect_cell)

    def basis_der(self, i, d):
        # The same of basis! d is an integer here
        raise NotImplementedError

    def _basis_der(self, i, d):
        # The same comment of _basis applies also here
        assert i < self._base_shape, "Wrong base dimension!"
        assert len(d) == 1, "This is a monodimensional space!"        
        func = self.basis_der(i[0],d[0])
        borders = self.basis_span(i[0])
        rect_cell = Rectangle(np.array((borders[0],)), 
                              np.array((borders[1],)))
        return FWithDomain(func, rect_cell)

    def eval(self, c, x, broadcast=-1):
        assert broadcast==-1, \
            "In the monodimensional vector space, the broadcast must be -1"
        MVS=MonodimensionalVectorSpace
        return super(MVS, self).eval(c, x, broadcast=-1)


    def eval_der(self, c, d, x, broadcast=-1):
        assert broadcast==-1, \
            "In the monodimensional vector space, the broadcast must be -1"
        MVS=MonodimensionalVectorSpace
        if not isinstance(d,Iterable):
            d = (d,)
        return super(MVS, self).eval_der(c, d, x, broadcast=-1)

    def element(self, c):
        """VectorSpace.element(c): a callable function,
        representing sum(c[i] * basis[i]) """
        return lambda x: self.eval(c,x)

    def element_der(self, c, d):
        return lambda x: self.eval_der(c, d, x)        



    def basis_span(self, i):
        """VectorSpace.basis_span(i): a tuple indicating the start
        and end indices into the cells object where the i-th basis
        function is different from zero"""
        raise NotImplementedError

    def cell_span(self, i):
        """ An array of indices containing the basis functions which
        are non zero on the i-th cell """
        raise NotImplementedError

    def print_info(self):
        print '============================================================'
        print 'Name: '+type(self).__name__
        print 'N dofs: {}, N cells: {}, \nCell boundaries: {}'.format(self.n_dofs, self.n_cells, self.cells)
        print 'Shared dofs on cell boundaries: {}'.format(self.n_dofs_per_end_point)
        for i in xrange(self.n_cells):
            print '------------------------------------------------------------'
            print 'Cell {}: [{},{}]'.format(i, self.cells[i], self.cells[i+1])
            print 'Nonzero basis: {}'.format(self.cell_span(i))
        print '------------------------------------------------------------'

    def __imul__(self, n):
        """This allows the construction of repeated VectorSpaces by simply
        multiplying the VectorSpace with an integer. """
        return RepeatedVectorSpace(self, n)


class AffineVectorSpace(MonodimensionalVectorSpace):
    """Affine transformation of a VectorSpace. Given a vector space, returns
    its affine transformation between a and b"""
    def __init__(self, vs, a=0.0, b=1.0):
        self.vs = vs
        a0 = vs.cells[0]
        b0 = vs.cells[-1]
        self.J = (b-a)/(b0-a0)
        cells = (vs.cells-a0)/(b0-a0)*(b-a) + a
        super(AffineVectorSpace, self).__init__(cells, vs.n_dofs,
                                                vs.n_dofs_per_end_point)
        self.a0 = a0
        self.b0 = b0
        self.a = a
        self.b = b
        try:
            self.degree = vs.degree
        except:
            pass

    def reset(self, a=0.0, b=1.0):
        """Make the affine transformation on the new [a,b] interval."""
        a0 = self.a0
        b0 = self.b0
        self.a = a
        self.b = b
        self.J = (b-a)/(b0-a0)
        self.cells = (self.vs.cells-a0)*self.J + a
        
    def pull_back(self, x):
        """Transform x from [a,b] to [a0, b0]"""
        return (x-self.a)/self.J+self.a0
        
    def push_forward(self, x0):
        """Transform x from [a0,b0] to [a, b]"""
        return (x0-self.a0)*self.J+self.a
    
    def basis(self, i):
        return lambda x: self.vs.basis(i)(self.pull_back(x))

    def basis_der(self, i, d):
        return lambda x: self.vs.basis_der(i,d)(self.pull_back(x))/(self.J**d)
 
    def basis_span(self, i):
        """VectorSpace.basis_span(i): a tuple indicating the start and end indices
        into the cells object where the i-th basis function is different from
        zero """
        return self.vs.basis_span(i)

    def cell_span(self, i):
        """ An array of indices containing the basis functions which are non zero on the i-th cell """
        return self.vs.cell_span(i)




class IteratedVectorSpace(MonodimensionalVectorSpace):
    """Construct an iterated version of the given original VectorSpace """
    def __init__(self, vsext, span):
        assert len(span) > 1, \
          'Expecting span to have at least two entries! Found {}'.format(len(span))
          
        self.vs = AffineVectorSpace(vsext, 0, 1)
        vs = self.vs
        self.span = span
        # Start by assuming discontinuous functions
        n_dofs = vs.n_dofs*(len(span)-1)
        n_dofs_per_end_point = vs.n_dofs_per_end_point
        if n_dofs_per_end_point > 0:
            n_dofs -= (len(span)-2)*vs.n_dofs_per_end_point
        n_cells = vs.n_cells*(len(span)-1)

        # All cells have already been set to 0,1. Remove the last
        # element, which is repeated
        loc_cells = vs.cells[0:-1]
        cells = np.array([])
        for i in xrange(len(span)-1):
            cells = np.append(cells, loc_cells*(span[i+1]-span[i])+span[i])
        cells = np.append(cells, [span[-1]])
        assert len(cells)-1 == n_cells, \
          "Internal error! {} != {}".format(len(cells)-1, n_cells)
          
        super(IteratedVectorSpace, self).__init__(cells, n_dofs,
                                                  n_dofs_per_end_point)
        try:  
            self.degree = vs.degree
        except:
            pass

    def local_to_global(self, base, i):
        """Return global index of ith local dof on cell c"""
        self.vs.check_index(i)
        # Keep track of the fact that the last n_dofs_per_end_point
        # are shared between consecutive basis
        return self.vs.n_dofs*base+i-base*self.n_dofs_per_end_point
        

    def global_to_local(self, i):
        """Given a global index, return the local base index (or indices), and the
        local dof (or dofs) to which this degree of freedom refers to as
        pairs of [ (local_vspace_index, local_index),
        (local_vspace_index, local_index)].  This has nothing to do with
        the cells indices. Those are returned by basis_span.
        """
        assert self.vs.n_dofs-self.n_dofs_per_end_point > 0, \
          'Internal error! {} ! > 0'.format(self.vs.n_dofs-self.n_dofs_per_end_point) 

        n_unique = (self.vs.n_dofs-self.n_dofs_per_end_point)
        b = int(np.floor(i/n_unique))
        j = np.mod(i, n_unique)
        ret = []
        if j < self.n_dofs_per_end_point and b>0:
            ret += [(b-1, self.vs.n_dofs-(self.n_dofs_per_end_point-j))]
        if b< len(self.span)-1:
            ret += [(b,j)]
        return ret

    def eval_basis(self, i, xx):
        self.check_index(i)
        pairs = self.global_to_local(i)
        x = np.squeeze(xx)
        y = np.squeeze(np.array(x*0))
        span = self.span
        for p in pairs:
            a = span[p[0]]
            b = span[p[0]+1]
            vs = self.vs
            vs.reset(a, b)
            if b == self.cells[-1]:
                b += 1
            ids = np.array( (a<=x) & (x<b) )
            print ids
            print p[1]
            print x[ids]
            y[ids] = vs.basis(p[1])(x[ids])
        return y
            
            
    def eval_basis_der(self, i, d, xx):
        self.check_index(i)
        pairs = self.global_to_local(i)
        x = np.squeeze(xx)
        y = np.squeeze(np.array(x*0))
        span = self.span
        for p in pairs:
            a = span[p[0]]
            b = span[p[0]+1]
            vs = self.vs
            vs.reset(a, b)
            if b == self.cells[-1]:
                b += 1
            ids = np.array( (a<=x) & (x<b) )
            y[ids] = vs.basis_der(p[1], d)(x[ids])
        return y
    
    def basis(self, i):
        return lambda x: self.eval_basis(i,x)

    def basis_der(self, i, d):
        return lambda x: self.eval_basis_der(i,d,x)

    def basis_span(self, i):
        """VectorSpace.basis_span(i): a tuple indicating the start and end indices
        into the cells object where the i-th basis function is different from
        zero
        """
        self.check_index(i)
        nbasis = len(self.span)-1
        pairs = self.global_to_local(i)
        start = self.vs.n_cells*pairs[0][0]+self.vs.basis_span(pairs[0][1])[0]
        end   = self.vs.n_cells*pairs[-1][0]+self.vs.basis_span(pairs[-1][1])[1]
        return (start, end)

    def cell_span(self, i):
        """ An array of indices containing the basis functions which are non zero on the i-th cell """
        b = i/self.vs.n_cells
        j = np.mod(i, self.vs.n_cells)
        startid = b*(self.vs.n_dofs-self.vs.n_dofs_per_end_point)
        local_ids = self.vs.cell_span(j)
        return [id+startid for id in local_ids]

class RepeatedVectorSpace(IteratedVectorSpace):
    """Construct an IteratedVectorSpace with uniform repetitions."""
    def __init__(self, vs, n):
        super(RepeatedVectorSpace, self).__init__(vs, np.linspace(0,1,n+1))


class DummyVectorSpace(MonodimensionalVectorSpace):
    """A VectorSpace just one element in the base: the function 1 in
    the interval 0 and 1. This class is usefull as an example and for
    some tests"""
    
    def __init__(self):
        super(DummyVectorSpace, self).__init__(np.array((0,1)), 1)
    
    def basis(self, i):
        return lambda x: 1

    def basis_der(self, i ,d):
        return lambda x : 0

    def basis_span(self, i):
        return (0,1)

    def cell_span(self, i):
        return [0]    


class DoubleLineVectorSpace(VectorSpace):
    """This is an example of an implementation of a 2D VectorSpace; it
    rappresents all the surfaces that are a line if restricted to any
    line that is parallel to one axis"""
    
    def __init__(self):
        super(DoubleLineVectorSpace, self).__init__(4, (2,2), 2)
        self._square = Rectangle(np.array((0.,0.)), np.array((1., 1.)))

    def _basis(self, i):
        assert i < self._base_shape
        if i == (0,0):
            def func(x):
                return 1
        if i == (1,0):
            def func(x):
                return x[0]
        if i == (0,1):
            def func(x):
                return x[1]
        if i == (1,1):
            def func(x):
                return x[0]*x[1]
        return FWithDomain(func, self._square)

    def _basis_der(self, i, d):
        assert i< self._base_shape
        assert len(d)==2
        new_index = tuple([i[j] - d[j] for j in range(2)])
        if new_index[0]<0 or new_index[1]<0:
            return FWithDomain(lambda x: 0, self._square)
        else:
            return self._basis(new_index)

          
