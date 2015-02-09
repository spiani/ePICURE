from nose.tools import assert_raises

from interfaces.f_with_domain import *
from interfaces.cell import Rectangle
from numpy import array, array_equal, sum

def test_FWithDomain_wrong_constructions():
    R = Rectangle(array([1,2]), array([10,10]))
    f = lambda x: x*x
    assert_raises(AssertionError, FWithDomain, "test", R)
    assert_raises(AssertionError, FWithDomain, f, "test")
    F = FWithDomain(f,R)
    assert array_equal(F(array([2,2])),array([4,4]))

def test_FWithDomain_domain_use():
    R = Rectangle(array([1,2]), array([10,10]))
    f = lambda x: x*x
    F = FWithDomain(f,R)
    a = array([1., 1.])
    assert_raises(OutOfDomainError, F, a, raise_error=True)
    assert F(a, raise_error=False) == 0
    assert F.is_inside_domain(a)==False

def test_FWithDomain_broadcast():
    I = Rectangle(array([2]),array([5]))
    R = Rectangle(array([1,2]), array([10,10]))
    f = lambda x: sum(x)
    a = array([[1., 1.],[5.,5.]])    
    F1 = FWithDomain(f,R)
    assert array_equal(F1(a, broadcast=1), (0, 10))
    F2 = FWithDomain(f,I)
    assert array_equal(F2(a, broadcast=-1), ((0, 0),(5,5)))
    

    
    

