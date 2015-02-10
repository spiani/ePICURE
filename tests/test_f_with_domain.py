from nose.tools import assert_raises

from interfaces.f_with_domain import *
from interfaces.cell import Rectangle
from numpy import array, array_equal, sum, power

def test_FWithDomain_wrong_constructions():
    R = Rectangle(array([1,2]), array([10,10]))
    def norm2(x):
        return power(sum(x*x, axis=-1), .5)
    assert_raises(AssertionError, FWithDomain, "test", R)
    assert_raises(AssertionError, FWithDomain, norm2, "test")
    F = FWithDomain(norm2,R)
    assert F(array([3,4])) == 5

def test_FWithDomain_domain_use():
    R = Rectangle(array([1,2]), array([10,10]))
    f = lambda x: sum(x, axis=-1)
    F = FWithDomain(f,R)
    a = array([1., 1.])
    print F(a)
    assert F(a) == 0
    assert F.is_inside_domain(a)==False

def test_FWithDomain_broadcast():
    R = Rectangle(array([1,2]), array([10,10]))
    f = lambda x: sum(x, axis=-1) +3 
    a = array(( ([1.1,2.1], [3,6]), 
                ([1,0], [1.1,9.6]) ))    
    F1 = FWithDomain(f,R)
    print F1(a)
    assert array_equal(F1(a), ((6.2,12.),(0.,13.7)))

def test_FWithDomain_broadcast_1D():
    R = Rectangle(2, 5)
    f = lambda x: power(x,2)
    a = array(( ([0, -1],  [2, 3]), 
                ([4, 5],  [6, -7])  ))    
    b = array(( ([0, 0 ], [4, 9]), 
                ([16,25], [0, 0])  ))    
    F1 = FWithDomain(f,R)
    print F1(a)
    assert array_equal(F1(a), b)

def test_FWithDomain_broadcast_1D_overshape():
    R = Rectangle(2, 5)
    f = lambda x: power(x,2)
    a = array(( ([[0], [-1]],  [[2], [3]]), 
                ([[4], [5]],  [[6], [-7]])  ))    
    b = array(( ([0, 0 ], [4, 9]), 
                ([16,25], [0, 0])  ))    
    F1 = FWithDomain(f,R)
    print F1(a)
    assert array_equal(F1(a), b)


def test_FWithDomain_broadcast_1D_all_zero():
    R = Rectangle(2, 5)
    f = lambda x: power(x,2)
    a = array(( ([0, -1],  [1, 7]), 
                ([1.198, 9.3],  [6, -7])  ))    
    b = array(( ([0, 0 ], [0, 0]), 
                ([0,0], [0, 0])  ))    
    F1 = FWithDomain(f,R)
    print F1(a)
    assert array_equal(F1(a), b)

def test_FWithDomain_broadcast_1D_all_zero_but_one():
    R = Rectangle(2, 5)
    f = lambda x: power(x,2)
    a = array(( ([0, -1],  [3, 7]), 
                ([1.198, 9.3],  [6, -7])  ))    
    b = array(( ([0, 0 ], [9, 0]), 
                ([0,0], [0, 0])  ))    
    F1 = FWithDomain(f,R)
    print F1(a)
    assert array_equal(F1(a), b)
    
def test_FWithDomain_broadcast_all_zero():
    R = Rectangle(array([1,2]), array([10,10]))
    f = lambda x: sum(x, axis=-1) +3 
    a = array(( ([1,1], [0,6]), 
                ([2,11], [3.2,0.4]) ))    
    F1 = FWithDomain(f,R)
    print F1(a)
    assert array_equal(F1(a), ((0,0),(0,0)))

def test_FWithDomain_broadcast_all_zero_but_one():
    R = Rectangle(array([1,2]), array([10,10]))
    f = lambda x: sum(x, axis=-1) +3 
    a = array(( ([1,1], [3,3]), 
                ([2,11], [3.2,0.4]) ))    
    F1 = FWithDomain(f,R)
    print F1(a)
    assert array_equal(F1(a), ((0,9),(0,0)))

    
    

