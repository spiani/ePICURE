from interfaces.cell import Cell
from collections import Iterable
from numpy import (ndarray, array, zeros, repeat, logical_not, expand_dims,
                   squeeze, nonzero, prod)
import numpy.ma as ma
from numpy.random import rand

class OutOfDomainError(Exception):
    # An exception to make the program aware that something is trying
    # to evaluate a function outside its domain
    pass

class FWithDomain(object):
    """ FWithDomain is a class that reppresent a function with its related
    domain. It is created from a tuple of two objects: a callable one that
    is the function and a cell that is the domain. When you call this 
    object passing a value, the object checks if the value is inside the
    cell; if it is not, it returns 0. Otherwise, the value is passed to
    the function and its output is returned. 
    For the current implementation, f must get an array of whatever shape,
    but the last dimension should be equal to the space dimension. In that
    case, f shall return an array of the same shape of the input value
    but without the last index, and in every place should be stored the
    result of the computation. See for example a sum function in R2:
    lambda x : numpy.sum(x, axis=-1)
    or the function that is 42 on every point
    lambda x: 42*numpy.ones(x.shape[:-1])
    and the norm function
    lambda x : numpy.sum(x*x, axis=-1)
    """
    def __init__(self, a_function, a_cell):
        assert isinstance(a_cell, Cell), \
          "The second argument should be an instance of the Cell class"
        assert hasattr(a_function,"__call__"),  \
          " The first argument should be a callable object"
        self._function = a_function
        self._cell = a_cell
        self._vector_capable = self._check_if_f_is_vector_capable()
    
    def is_inside_domain(self, x):
        """ Check if x is inside the domain"""
        return self._cell.is_inside(x)
        
    def _check_if_f_is_vector_capable(self):
        # This function checks if the function supports vectorization
        # by itself. If not, it will be more cautious while passing
        # output
        
        dim = self._cell.get_space_dimension()
        
        # Output shall not be a vector if there is just one argument
        if dim==1:
            if isinstance(self._function(.99),Iterable):
                return False
        else:
            test = rand(dim)
            if isinstance(self._function(test),Iterable):
                return False
        
        # If there are more than one argument, the shape of the output
        # must be the same
        if dim==1:
            test = rand(3,5,2)
        else:
            test = rand(3,5,2,dim)
        test_result = self._function(test)
        if (not isinstance(self._function(test),ndarray) 
                                   or test_result.shape != (3,5,2)):
            return False
        
        # if all test are passed
        return True

    def __call__(self, x):
        dim = self._cell.get_space_dimension()
        # If the input is just a 1D value
        if not isinstance(x,ndarray):
            if self.is_inside_domain(x):
                return self._function(x)
            return 0.

        if dim == 1 and x.shape[-1] != 1:
            x = expand_dims(x, -1)

        # If is just one single evaluation of a vector of dimension dim
        if x.shape == (dim,):
            if self.is_inside_domain(x):
                return self._function(x)
            return 0.
 
        if self._vector_capable:
            return self._fast_call(x)
        else:
            return self._safe_call(x)

    def _fast_call(self, x):
        dim = self._cell.get_space_dimension()
       
        # Because of the different behaviour of the scalar and vector functions,
        # it is worth to examine every case by itself
        if dim == 1:
            while x.shape[-1] == 1:
                x = x.squeeze(-1)
            repetitions = prod(x.shape)
            old_shape = x.shape
            output = zeros(repetitions)
            x = x.flatten()
            good_values = self.is_inside_domain(x)
            good_val_where = nonzero(good_values)[0]
            good_val_n = good_val_where.size
            if good_val_n != 0:
                mask = logical_not(good_values)
                x_mask = ma.array(x, mask=mask)
                to_be_evaluated = x_mask.compressed()
                f_eval = self._function(to_be_evaluated)
                for i in xrange(good_val_n):
                    output[good_val_where[i]] = f_eval[i]
            output = output.reshape(old_shape)
            return output

 
        if x.shape[-1] == dim:
            repetitions = prod(x.shape)/dim            
            old_shape = x.shape
            output = zeros(repetitions)
            x = x.reshape((repetitions, dim))
            good_values = self.is_inside_domain(x)
            good_val_where = nonzero(good_values)[0]
            good_val_n = good_val_where.size
            if good_val_n != 0:
                mask = logical_not(good_values)
                x_mask = ma.array(x, mask=repeat(mask, dim, axis=-1))
                to_be_evaluated = x_mask.compressed().reshape((good_val_n,dim))
                f_eval = self._function(to_be_evaluated)
                for i in xrange(good_val_n):
                    output[good_val_where[i]] = f_eval[i]
            output = output.reshape(old_shape[:-1])
            return output
        
        raise ValueError, "The argument is in an incorrect format"


    def _safe_call(self, x):
        dim = self._cell.get_space_dimension()
        if dim == 1:
            while x.shape[-1] == 1:
                x = x.squeeze(-1)
            repetitions = prod(x.shape)
            old_shape = x.shape
            output = zeros(repetitions)
            x = x.flatten()
            for i in xrange(repetitions):
                if self.is_inside_domain(x[i]):
                    output[i] = self._function(x[i])
            output = output.reshape(old_shape)
            return output
        else:
            if x.shape[-1] != dim:
                raise ValueError("Space has dimension " + str(dim) + "but the"
                                 " argument has dimension " +str(x.shape[-1]))
            repetitions = prod(x.shape[-1])
            new_shape = x.shape[:-1]
            output = zeros(repetitions)
            x = x.reshape((repetitions,dim))
            for i in xrange(repetitions):
                if self.is_inside_domain(x[i]):
                    output[i] = self._function(x[i])
            output = output.reshape(new_shape)
            return output
            

        
