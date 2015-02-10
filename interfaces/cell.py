import numpy as np
from collections import Iterable

def ensure_array(P):
    """Ensure that a value is a numpy array; if it is any other sort
    of iterable, it is changed in a numpy array; if it is a not an
    iterable object, it is wrapped inside a numpy array"""
    if not isinstance(P, np.ndarray):
        if not isinstance(P, Iterable):
            P = np.array((P,))
        else:
            P = np.array(P)
    return P

class Cell(object):
    """A cell is a subset of Rn that has just a pourpose: tell if
    an array is or not inside the cell"""
    def is_inside(self, P):
        """Return True if the point P is inside the cell, False
        otherwise"""
        raise NotImplementedError
        
    def get_space_dimension(self):
        """Return the dimension of the space where the cell
        lives"""
        raise NotImplementedError        

    def __mul__(self, a_cell):
        assert isinstance(a_cell, Cell), \
           "I can NOT multiply a cell by something different than a cell"        
        # This is a little bit tricky! If the second element of the
        # multiplication is a product_cell, then the special product
        # function of that class is called
        if isinstance(a_cell, CellProduct):
            return a_cell * self
        
        return CellProduct(self, a_cell)



class Rectangle(Cell):
    """This cell is a rectangle in the Rn space. The points
    lb_corner and rt_corner are the two corners of the rectangle:
    lb_corner is the left bottom corner, rt_corner is the right
    top one. The border is also part of the cell"""
    
    def __init__(self, lb_corner, rt_corner):
        lb_corner = ensure_array(lb_corner)
        rt_corner = ensure_array(rt_corner)
        assert lb_corner.shape == rt_corner.shape, \
            "lb_corner and rt_corner should have the same dimension"
        assert lb_corner.shape == (len(lb_corner),), \
                      "lb_corner has more than one dimension!"
        self._lb_corner = lb_corner 
        self._rt_corner = rt_corner
        
    def get_corners(self):
        return (self._lb_corner, self._rt_corner)
    
    def get_space_dimension(self):
        return len(self._lb_corner)
    
    def is_inside(self, P):
        P = ensure_array(P)
        if self.get_space_dimension() == 1 and P.shape[-1] != 1:
            P = np.expand_dims(P,-1)
        assert P.shape[-1] == self._lb_corner.shape[0], \
                      "P dimension is wrong (" + str(P.shape[-1]) +")!"                        

        return np.all(P>=self._lb_corner, axis=-1) * np.all(P<=self._rt_corner, axis=-1)
    

class CellProduct(Cell):
    """A cell product is a new cell in space R(n+m) constructed starting
    from two (or more) cells in Rn and Rm. A point P=[a1,...,an,b1,...,bm]
    is in the product cell if and only if P1=[a1,...,an] is in the first
    cell and P2=[b1,...,bm] is in the second one"""
    
    def __init__(self, *args):
        # Check if the input is just one iterable object
        # or the cells are passed one by one
        if len(args)==1:
            assert (not isinstance(args[0],Cell)), \
              'It is impossible to create a cell product starting from just one cell'
            cell_list = args[0]
        else:
            cell_list = list(args)
        
        assert len(cell_list)==len([i for i in cell_list if isinstance(i,Cell)]), \
            'All the input values should be cells that will be multiply togheter'
        
        self._cell_list = cell_list
    
    def get_space_dimension(self):
        return sum(i.get_space_dimension() for i in self._cell_list)
        
    def get_base_cells(self):
        """Return a list of all the cell that are multiplied inside this cell product"""
        return self._cell_list[:]
        
    def get_base_dimensions(self):
        """Return the dimensions of the spaces of the base cells of this product"""
        return tuple(i.get_space_dimension() for i in self._cell_list)

    def is_inside(self, *args):
        # This function can handle both a tuple of two points of
        # dimension n and m or one point of dimension m+n
        assert len(args) in [1,len(self._cell_list)], \
                "Wrong number of arguments (only 1 point or a tuple of point for every cell)!"
        # In this case the argument are passed as a list or just as
        # one vector
        if len(args)==1:
            # If the arguments are inside a tuple, expand them and
            # call again the function
            if isinstance(args[0],(tuple,list)):
                return self.is_inside(*args[0])
            # Here I expect a numpy vector, I divide it and I check
            # for every "piece" if it is inside the right cell
            else:
                point = args[0]
                assert point.shape == (self.get_space_dimension(),), \
                    "The point should be a flat vector of dimension m+n (or a touple of points)"
                temp_dim = 0
                for i, current_cell in enumerate(self._cell_list):
                    current_dim = current_cell.get_space_dimension()
                    current_point = point[temp_dim : temp_dim+current_dim]
                    if not current_cell.is_inside(current_point):
                        return False
                    temp_dim += current_dim
        # In this case the args are expanded
        else:
            for i, current_cell in enumerate(self._cell_list):
                current_point = args[i]
                if not current_cell.is_inside(current_point):
                    return False
        return True

    def __mul__(self, a_cell):
        assert isinstance(a_cell, Cell), \
           "I can NOT multiply a cell by something different than a cell"
        
        cell_list = self._cell_list[:]

        if isinstance(a_cell, CellProduct):
            cell_list.extend(a_cell.get_base_cells())
            return CellProduct(cell_list)

        cell_list.append(a_cell)
        return CellProduct(cell_list)
