from nose.tools import assert_raises

from interfaces.cell import *
from numpy import array, zeros, array_equal
from numpy.random import rand, seed, randint


def test_cell_interface():
    c = Cell()
    assert_raises(NotImplementedError, c.is_inside, None)
    assert_raises(NotImplementedError, c.get_space_dimension)

def test_cell_rectangle2D():
    lb_cor = array([-7,2])
    rt_cor = array([2,5])
    R = Rectangle(lb_cor, rt_cor)
    assert R.get_space_dimension() == 2

def test_cell_rectangleND():
    seed(0)
    for i in range(2,100):
        lb_cor = rand(i)
        rt_cor = rand(i) + lb_cor
        R = Rectangle(lb_cor, rt_cor)
        assert R.get_space_dimension() == i
    
def test_cell_rectangle_wrong_dim1():
    lb_cor = array([-7., 2., 1.])
    rt_cor = array([2., 5.])
    assert_raises(AssertionError, Rectangle, lb_cor, rt_cor)

def test_cell_rectangle_wrong_dim2():
    lb_cor = array([-7.3, 2.1])
    rt_cor = array([2., 5.3, 1.5])
    assert_raises(AssertionError, Rectangle, lb_cor, rt_cor)

def test_cell_rectangle_get_corners():
    seed(0)
    for i in range(2,100):
        lb_cor = rand(i)
        rt_cor = rand(i) + lb_cor
        R = Rectangle(lb_cor, rt_cor)
        assert R.get_corners()[0] is lb_cor
        assert R.get_corners()[1] is rt_cor
        
def test_cell_rectangle_is_inside():
    seed(0)
    for i in range(2,100):
        lb_cor = rand(i)
        rt_cor = rand(i) + lb_cor
        R = Rectangle(lb_cor, rt_cor)
        point1 = .5 * (lb_cor + rt_cor)
        assert R.is_inside(point1) == True
        for j in range(i):
            error = zeros((i,))
            error[j] = 0.0001
            point2 = lb_cor - error
            point3 = rt_cor + error
            assert R.is_inside(point2) == False
            assert R.is_inside(point3) == False
        
def test_cell_rectangle_border_inside():
    seed(0)
    for i in range(1,10):
        lb_cor = rand(i)
        diff = rand(i)
        rt_cor = diff + lb_cor
        R = Rectangle(lb_cor, rt_cor)
        point = lb_cor + diff*randint(0,2,i)
        assert R.is_inside(point) == True

def test_cell_rectangle_broadcast():
    lb_cor = array([-7,2,0])
    rt_cor = array([2,5,1])
    a = array([[(1,3,1),(1,1,1),(0,3,0)],[(3,3,0),(0.1,2.1,.1),(3,1,.1)]])
    R = Rectangle(lb_cor, rt_cor)
    assert array_equal(R.is_inside(a), array([[True, False, True], [False, True, False]]))
    

def test_cell_product1():
    R1 = Rectangle(array([1,1]), array([2,2]))
    R2 = Rectangle(array([.5, .5, .5]), array([1.5, 1.5, 1.5]))
    R3 = Rectangle(array([1]),array([2]))
    T = R1 * R2 * R3
    p =  array([1.5, 1.5, 1, 1, 1, 1.5])
    p1 = array([1.5, 1.5])
    p2 = array([1, 1, 1])
    p3 = array([1.5])
    assert T.is_inside(p) == True
    assert T.is_inside([p1,p2,p3]) == True
    assert T.is_inside((p1,p2,p3)) == True
    assert T.is_inside(p1,p2,p3) == True
    pe =  array([1.5, 1.5, 3, 1, 1, 1.5])
    p1e = array([2.5, 1.5])
    p2e = array([1, 2, 1])
    p3e = array([2.5])
    assert T.is_inside(pe) == False
    assert T.is_inside([p1e,p2,p3]) == False
    assert T.is_inside([p1,p2e,p3]) == False
    assert T.is_inside([p1e,p2,p3e]) == False
    assert T.is_inside((p1e,p2e,p3e)) == False
    assert T.is_inside(p1,p2,p3e) == False

def test_cell_product_among_products():
    R1 = Rectangle(array([1,1]), array([2,2]))
    R2 = Rectangle(array([.5, .5, .5]), array([1.5, 1.5, 1.5]))
    R3 = Rectangle(array([1]),array([2]))
    T1 = R1 * R2
    T2 = R1 * R3
    T = T1 * T2
    assert T.get_base_dimensions() == (2,3,2,1)

def test_reverse_product():
    R1 = Rectangle(array([1,1]), array([2,2]))
    R2 = Rectangle(array([.5, .5, .5]), array([1.5, 1.5, 1.5]))
    R3 = Rectangle(array([1]),array([2]))
    T1 = R2 * R3
    T = R1 * T1
    assert len(T.get_base_cells()) == 3
    assert len(T.get_base_dimensions()) == 3
