from interfaces.cell import Cell
from collections import Iterable
from numpy import array, prod, ndarray, unravel_index

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
    While calling the objects, the broadcasting parameter could be setted
    (default is broadcast=0). This means that the argument of the function
    is an array and that its values should be passed to the fanction. To be
    more clear, this is what will happen:
    - if broadcast = 0 the function is executed on x
    - if broadcast = 1 the function is executed on every x[i]
    - if broadcast = 2 the function is executed on every x[i,j]
    - ...
    - if broadcast = -1 the function is executed on the last possible
      elements of x
    If you want, you can also set the "return_error=True" option. In that
    case, if the point is not inside the domain, the function will rise an
    error. Be aware that this option is automatically disabled if broadcast
    is not 0."""
    def __init__(self, a_function, a_cell):
        assert isinstance(a_cell, Cell), \
          "The second argument should be an instance of the Cell class"
        assert hasattr(a_function,"__call__"),  \
          " The first argument should be a callable object"
        self._function = a_function
        self._cell = a_cell
    
    def is_inside_domain(self, x):
        """ Check if x is inside the domain"""
        return self._cell.is_inside(x)
    
    def __call__(self, x, broadcast=0, raise_error=False):
        
        # Let start without broadcast
        no_br = (broadcast==0 or (broadcast==-1 and not isinstance(x,Iterable)))
        if no_br:
            if not self.is_inside_domain(x):
                if raise_error:
                    raise OutOfDomainError
                return 0
            else:
                return self._function(x)
        
        # To use broadcast, we need arrays
        assert isinstance(x,ndarray), \
                   "x must be a numpy array if you want to use broadcast!"
        # Now the maximum level
        if broadcast == -1 or broadcast>= len(x.shape):
            def eval_func(i):
                indx = unravel_index(i,x.shape)
                if self.is_inside_domain(x[indx]):
                    return self._function(x[indx])
                return 0
            output = array(map(eval_func,xrange(x.size))).reshape(x.shape) 
            return output
          
        # An intermediate broadcast level
        variating_indices = x.shape[:broadcast]
        total_exec = prod(variating_indices)
        def eval_func(i):
            indx = unravel_index(i,variating_indices)
            if self.is_inside_domain(x[indx]):
                return self._function(x[indx])
            return 0
        output = array(map(eval_func,xrange(total_exec)))
        if isinstance(output[0],ndarray):
            variating_indices += output[0].shape
        return output.reshape(variating_indices) 
            
        
