import numpy as np
import time as tm

from .rot import QuatRotator
from .rot import QuatRotator2
from .rot import MatrixRotator
from .rot import ERRotator
from .rot import QERRotator
from .rot import Rotator
from .rot import xyz2q, zyx2q, q2w, w2xyz, w2zyx
from .rot import A2xyz, A2zyx, w2A, q2A, xyz2q, zyx2q, zyx2w, xyz2w
from .rot import q2u, u2q, w2u, u2w, w2q, q2w
from .rot import w2q_, q2w_, q2w__
from .rot import wv2rt, wv2tr, wchain

from numpy import linalg as LA

import quaternion as quat

rotations = (QuatRotator, QuatRotator2, MatrixRotator, ERRotator,QERRotator)

def test1(random = False, time = None):
    """
    test single rotations
    """
    if random:
        omega = np.random.rand(3)
    else:
        omega = np.array([0.,0.,1.])
    vec = np.array([1.,2.,3.])
    results = []
    for r in rotations:
        x = r(omega, time)(vec)
        print(f'{r}\n{x}')
        results.append(x)
    results = np.asarray(results)
    print('\nResiduals:')
    r = np.std(results, axis=0)/np.mean(results, axis=0)
    print(f'{r}')
    t = np.sqrt(np.average(r*r))
    print(f'\n RESIDUAL: {t}')

def test2(random = False, dims = 1, time = 1.):
    """
    test rotation of vector array
    """
    if random:
        omega = np.random.rand(3)
    else:
        omega = np.array([0.,0.,1.])
    if dims == 0:
        vec = np.array([1.,2.,3.])
    if dims == 1:
        vec = np.array([[1.,2.,3.], [4.,5.,6.]])
    else:
        vec = np.array([[[1.,2.,3.], [4.,5.,6.]],[[1.,2.,4.], [4.,5.,7.]]])
    results = []
    for r in rotations:
        x = r(omega, time)(vec)
        print(f'{r}\n{x}')
        results.append(x)
    results = np.asarray(results)
    print('\nResiduals:')
    r = np.std(results, axis=0)/np.mean(results, axis=0)
    print(f'{r}')
    t = np.sqrt(np.average(r*r))
    print(f'\n RESIDUAL: {t}')

def test3(random = False):
    """
    test time array
    """
    time = np.array([1., 2., 3., 4.])
    if random:
        omega = np.random.rand(3)
    else:
        omega = np.array([0.,0.,1.])
    vec = np.array([1.,2.,3.])
    results = []
    for r in rotations:
        x = r(omega, time)(vec)
        print(f'{r}\n{x}')
        results.append(x)
    results = np.asarray(results)
    print('\nResiduals:')
    r = np.std(results, axis=0)/np.mean(results, axis=0)
    print(f'{r}')
    t = np.sqrt(np.average(r*r))
    print(f'\n RESIDUAL: {t}')


def test4(random = False):
    """
    test vector array
    """
    #rotations = ()
    #rotations += (QuatRotator,)
    #rotations += (QuatRotator2)
    #rotations += (MatrixRotator,)
    #rotations += (ERRotator,)
    time = 1.
    if random:
        omega = np.random.rand(6).reshape((2,3))
    else:
        omega = np.array([[0.,0.,1.],[0.,1.,0.],[1.,0.,0.],[1.,1.,1.],[-1,1.,0]])
    vec = np.array([1.,2.,3.])
    results = []
    for r in rotations:
        x = r(omega, time)(vec)
        print(f'{r}\n{x}')
        results.append(x)
    results = np.asarray(results)
    print('\nResiduals:')
    r = np.std(results, axis=0)/np.mean(results, axis=0)
    print(f'{r}')
    t = np.sqrt(np.average(r*r))
    print(f'\n RESIDUAL: {t}')

def test5(random = False, time = True, omega = True, vec = True):
    """
    test many things
    """
    #rotations = ()
    #rotations += (QuatRotator,)
    #rotations += (QuatRotator2)
    #rotations += (MatrixRotator,)
    #rotations += (ERRotator,)
    if time is True:
        time = np.array([1., 2., 3., 4.])
    else:
        time = 1.
    if omega is True:
        if random:
            omega = np.random.rand(5*3).reshape((5,3))
        else:
            omega = np.array([[0.,0.,1.],[0.,1.,0.],[1.,0.,0.],[1.,1.,1.],[-1,1.,0]])
    else:
        if random:
            omega = np.random.rand(3)
        else:
            omega = np.array([0.,0.,1.])
    if vec is True:
        vec = np.array([[1.,2.,3.], [4.,5.,6.]])
    else:
        vec = np.array([1.,2.,3.])
    results = []
    for r in rotations:
        x = r(omega, time)(vec)
        print(f'\n{r}\n{x}')
        results.append(x)
    results = np.asarray(results)
    print('\nResiduals:')
    r = np.std(results, axis=0)/np.mean(results, axis=0)
    print(f'{r}')
    t = np.sqrt(np.average(r*r))
    print(f'\n RESIDUAL: {t}')


def test6(random = False, time = True, align = (5,)):
    """
    test align
    """
    #rotations = ()
    #rotations += (QuatRotator,)
    #rotations += (QuatRotator2,)
    #rotations += (MatrixRotator,)
    #rotations += (ERRotator,)
    if time is True:
        time = np.array([[.1, 1., 2., 3., 4.],
                         [5., 6., 7., 8., 9.]])
    else:
        time = 1.
    assert isinstance(align, tuple)
    omega = np.random.rand(np.product(align)*4*3).reshape((4,)+align+(3,))
    vec = np.arange(np.product(align)*4*6*3).reshape(align+(4,6,3))
    results = []
    for r in rotations:
        x = r(omega, time)(vec, align = len(align))
        #print(f'\n{r}\n{x}')
        print(f'\n{r}')
        results.append(x)
    results = np.asarray(results)
    #print('\nResiduals:')
    r = np.std(results, axis=0)/np.mean(results, axis=0)
    #print(f'{r}')
    t = np.sqrt(np.average(r*r))
    print(f'\n RESIDUAL: {t}')

def test7(random = False, time = True, align = (5,)):
    """
    test phase
    """
    #rotations = ()
    #rotations += (QuatRotator,)
    #rotations += (QuatRotator2,)
    #rotations += (MatrixRotator,)
    #rotations += (ERRotator,)
    if time is True:
        time = np.array([[.1, 1., 2., 3., 4.],
                         [5., 6., 7., 8., 9.]])
    else:
        time = 1.
    assert isinstance(align, tuple)
    omega = np.random.rand(np.product(align)*4*3).reshape((4,)+align+(3,))
    vec = np.arange(np.product(align)*4*6*3).reshape(align+(4,6,3))
    phase = np.random.rand(np.product(omega.shape[-2:-1])).reshape(omega.shape[-2:-1])
    results = []
    for r in rotations:
        x = r(omega, time, phase)(vec, align = len(align))
        #print(f'\n{r}\n{x}')
        print(f'\n{r}')
        results.append(x)
    results = np.asarray(results)
    #print('\nResiduals:')
    r = np.std(results, axis=0)/np.mean(results, axis=0)
    #print(f'{r}')
    t = np.sqrt(np.average(r*r))
    print(f'\n RESIDUAL: {t}')

def test_speed():
    """
    test align
    """
    rotations = ()
    rotations += (QuatRotator,)
    rotations += (QuatRotator2,)
    rotations += (MatrixRotator,)
    rotations += (QERRotator,)
    rotations += (ERRotator,)
    time = np.arange(1000).reshape(-1,10)/1000
    vec = np.random.rand(3000).reshape((10,10,-1,3))
    omega = np.random.rand(300).reshape(20,-1,3)
    for r in rotations:
        t0, p0 = tm.time(), tm.process_time()
        x = r(omega, time)(vec)
        print(f'\n{r} - {tm.process_time()-p0} s, {tm.time()-t0} s')

def test_xyz(n = 1000):
    w = 1 - 2 * np.random.rand(3*n).reshape((-1, 3))
    w = w / LA.norm(w, axis = -1)[..., np.newaxis]
    v = q2w(xyz2q(w2xyz(w)))
    u = q2w(xyz2q(A2xyz(w2A(w))))
    results = np.asarray([w, v, u])
    r = np.std(results, axis=0)/np.mean(results, axis=0)
    t = np.sqrt(np.average(r*r))
    print(f'\n RESIDUAL: {t}')

def test_zyx(n = 1000):
    w = 1 - 2 * np.random.rand(3*n).reshape((-1, 3))
    w = w / LA.norm(w, axis = -1)[..., np.newaxis]
    v = q2w(zyx2q(w2zyx(w)))
    u = q2w(zyx2q(A2zyx(w2A(w))))
    results = np.asarray([w, v, u])
    r = np.std(results, axis=0)/np.mean(results, axis=0)
    t = np.sqrt(np.average(r*r))
    print(f'\n RESIDUAL: {t}')


def test_qwu(n = 1000):
    w = 1 - 2 * np.random.rand(3*n).reshape((-1, 3))
    f = 1 - 2 * np.random.rand(4*n).reshape((-1, 4))
    q = quat.from_float_array(f / LA.norm(f, axis = -1)[..., np.newaxis])
    u = q2u(q)
    results = []
    results.append(q2u(u2q(u)) - u)
    results.append(w2u(u2w(u)) - u)
    results.append(quat.as_float_array(u2q(q2u(q)) - q))
    results.append(quat.as_float_array(w2q(q2w(q)) - q))
    results.append(u2w(w2u(w)) - w)
    results.append(q2w(w2q(w)) - w)
    t = [np.sqrt(np.average(r*r)) for r in results]
    print(f'\n RESIDUAL: {t}')

def test_qw_(n = 1000):
    w = 1 - 2 * np.random.rand(3*n).reshape((-1, 3))
    f = 1 - 2 * np.random.rand(4*n).reshape((-1, 4))
    q = quat.from_float_array(f / LA.norm(f, axis = -1)[..., np.newaxis])
    results = []
    results.append(q2w(w2q(w)) - w)
    results.append(q2w_(w2q(w)) - w)
    results.append(q2w__(w2q(w)) - w)
    results.append(q2w(w2q_(w)) - w)

    results.append(quat.as_float_array(w2q(q2w(q)) - q))
    results.append(quat.as_float_array(w2q_(q2w(q)) - q))
    results.append(quat.as_float_array(w2q(q2w_(q)) - q))
    results.append(quat.as_float_array(w2q(q2w__(q)) - q))

    t = [np.sqrt(np.average(r*r)) for r in results]
    print(f'\n RESIDUAL: {t}')


def test_wv2(n = 1000):
    w = 1 - 2 * np.random.rand(3*n).reshape((-1, 3))
    v = 1 - 2 * np.random.rand(3*n).reshape((-1, 3))
    results = []

    results.append(np.array([wchain(*wv2rt(wx, vx)) - wx for wx, vx in zip(w,v)]))
    results.append(np.array([wchain(*wv2tr(wx, vx)) - wx for wx, vx in zip(w,v)]))

    results.append(np.array([Rotator(wv2rt(wx, vx)[0])(vx) - Rotator(wx)(vx) for wx, vx in zip(w,v)]))
    results.append(np.array([Rotator(wv2tr(wx, vx)[1])(vx) - Rotator(wx)(vx) for wx, vx in zip(w,v)]))

    results.append(np.array([Rotator(wv2rt(wx, vx)[1])(Rotator(wx)(vx)) - Rotator(wx)(vx) for wx, vx in zip(w,v)]))
    results.append(np.array([Rotator(wv2tr(wx, vx)[0])(vx) - vx for wx, vx in zip(w,v)]))

    t = [np.sqrt(np.average(r*r)) for r in results]
    print(f'\n RESIDUAL: {t}')
