from interfaces.vector_space import *
from numpy import array
from numpy import allclose as eq

def test_vector_space_interface():
    vs = DoubleLineVectorSpace()
    assert vs.n_dofs == 4
    assert vs._base_shape == (2,2)
    c = array( ((1,2),
                (3,4))  )
    f = vs.element(c)
    assert f(array((0,0))) == 1
    assert f(array((1,0))) == 1+3
    assert f(array((0,1))) == 1+2
    assert f(array((1,1))) == 1+2+3+4

def test_vector_space_c_overshaped():
    vs = DoubleLineVectorSpace()
    c = array( ([(1,2), (2,4)],
                [(3,6), (4,8)])  )
    f = vs.element(c)
    assert eq( f(array((0,0))), (1,2))
    assert eq( f(array((1,0))), (1+3, 2+6))
    assert eq( f(array((0,1))), (1+2, 2+4))
    assert eq( f(array((1,1))), (1+2+3+4, 2+4+6+8))

def test_vector_space_c_overshaped_broadcast():
    vs = DoubleLineVectorSpace()
    c = array( ([(1,2), (2,4)],
                [(3,6), (4,8)])  )
    f = vs.element(c, broadcast=0)
    assert eq( f(array((1,0))), (1+3, 2+6))
    assert eq( f(array((0,1))), (1+2, 2+4))
    assert eq( f(array((1,1))), (1+2+3+4, 2+4+6+8))
    f = vs.element(c, broadcast=1)
    assert eq( f(array([(0,0),(1,1)])), ((1,10),(2,20)))



def test_vector_space_c_really_overshaped():
    vs = DoubleLineVectorSpace()
    c1 = array( ((1,2,5),
                 (3,4,6))  )
    c2 = array( ((.1,.2,.5),
                 (.3,.4,.6))  )
    c3 = array( ((.01,.02,.05),
                (.03,.04,.06))  )
    c4 = array( ((.001,.002,.005),
                (.003,.004,.006))  )
    c = array( ((c1, c2),
                (c3, c4)) )
    f = vs.element(c)
    assert(f(array([1,0])).shape == (2,3))
    a = array( ((1.111, 2.222,5.555),(3.333, 4.444, 6.666)) )
    assert(eq(f(array([1,1])),a)) 
    

def test_vector_space_c_overshaped_derivate():
    vs = DoubleLineVectorSpace()
    c = array( ([(1,2), (2,4)],
                [(3,6), (4,8)])  )
    f = vs.element_der(c,(1,1))
    assert eq( f(array((0,0))), (4, 8))
    assert eq( f(array((1,0))), (4, 8))
    assert eq( f(array((0,1))), (4, 8))
    assert eq( f(array((1,1))), (4, 8))
    f = vs.element_der(c,(0,0))
    g = vs.element(c)
    assert eq( f(array((.25,.25))), g(array((.25, .25))) )
    f = vs.element_der(c,(0,1))
    assert eq( f(array((0,0))), (2, 4))
    assert eq( f(array((1,0))), (6, 12))
    assert eq( f(array((0,1))), (2, 4))
    assert eq( f(array((1,1))), (6, 12))


def test_vector_space_c_really_overshaped_derivate():
    vs = DoubleLineVectorSpace()
    c1 = array( ((1,2,5),
                 (3,4,6))  )
    c2 = array( ((.1,.2,.5),
                 (.3,.4,.6))  )
    c3 = array( ((.01,.02,.05),
                (.03,.04,.06))  )
    c4 = array( ((.001,.002,.005),
                (.003,.004,.006))  )
    c = array( ((c1, c2),
                (c3, c4)) )
    f = vs.element_der(c, (1,0))
    assert(f(array([1,0])).shape == (2,3))
    a = array( ((0.011, 0.022, 0.055),(0.033, 0.044, 0.066)) )
    assert(eq(f(array([1,1])),a)) 
    
    

def test_monodim_vector_space_interface():
    a = DummyVectorSpace()
    assert a.n_dofs == 1
    assert a.n_cells == 1
    try:
        a.check_index(2)
        assert False, 'Expecting Failure!'
    except:
        pass
    assert a.basis(0)(.5) == 1
    
    try:
        a.basis(0)(1.2)
        assert False, 'Expecting Failure!'
    except:
        pass
    
