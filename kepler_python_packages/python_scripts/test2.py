import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.patches import Polygon

def Sum(v:'value', e:'error', w:'weight'=None):
    n = len(v)
    assert len(v) == len(e)    
    if w is None:
        w = np.array([1.]*len(v))
    else:
        assert len(w) == len(v)
    # wx = np.sum(w)
    w = np.array(w) / e**2
    wt = 1/np.sum(w)
    a = np.sum(v * w) * wt
    a2 = np.sum(v**2 * w) * wt
    # e2 = wt*wx + (a2 - a**2) * n/(n-1)
    e2 = wt + (a2 - a**2) * n/(n-1)
    return a, np.sqrt(e2)

def SumM(v:'value', e:'error', w:'weight'=None):
    """
    Meta Analysis

    p 690-695, ISBN 0-534-41820-1
    Fundamentals of Biostatistics, 6th eddition
    Bernard Rosner
    Thompson Higher Education, Belmont
    """

    v = np.array(v)
    e = np.array(e)

    n = len(v)
    assert len(v) == len(e)    
    if w is None:
        w = np.array([1.]*len(v))
    else:
        assert len(w) == len(v)
    w = np.array(w) / e**2
    wt = np.sum(w)
    w2t = np.sum(w**2)
    wti = 1/np.sum(w)
    yw = np.sum(w * v) * wti
    Qw = np.sum(w * (v - yw) ** 2)
    d2 = max(0, (Qw - (n-1)) / (wt - w2t*wti))
    wx = 1 / (e**2 + d2)
    wxti = 1 / np.sum(wx)
    a = np.sum(wx * v) * wxti
    e2 = wxti
    return a, np.sqrt(e2)



def f1():
    n = 12
    v = np.arange(n)*1
    e = np.arange(n)**0 * 1e-0
    # v = np.arange(n)**0.75 * 0.2
    # e = (np.arange(n)+1)**0.7 * 1e-1
    w = np.arange(n)*0+1
    a0,e0 = Sum(v,e,w)
    print('total:', a0,e0)
    j = n//3
    a1,e1 = Sum(v[:j],e[:j])
    a2,e2 = Sum(v[j:],e[j:])

    print('part1:',a1,e1)
    print('part2:',a2,e2)
    
    vx = np.array([a1,a2])
    ex = np.array([e1,e2])
    af,ef = Sum(vx,ex)
    print('combi:',af,ef)

    f = plt.figure()
    a = f.add_subplot(111)
    dx = 0.0001
    x = np.arange(-1,v[-1]+2,dx)
    c = ['b']*j + ['g']*(n-j)
    for i in range(n):
        a.plot(x,gaussian(x,v[i],e[i]),color=c[i])
    a.plot(x,gaussian(x,a1,e1),color=c[0],lw=2)
    a.plot(x,gaussian(x,a2,e2),color=c[-1],lw=2)
    a.plot(x,gaussian(x,af,ef),color='r',lw=2)
    a.plot(x,gaussian(x,a0,e0),color='y',lw=2)

    plt.draw()



def Sumb(v:'value', e:'error', w:'weight'=None):
    n = len(v)
    assert len(v) == len(e)    
    if w is None:
        w = np.array([1.]*len(v))
    else:
        assert len(w) == len(v)
    w = np.array(w) * e**(-2) 
    wt = 1/np.sum(w)
    a = np.sum(v * w) * wt
    e2 = wt # same result a Bayesian
    return a, np.sqrt(e2)

def gaussian(x, mu, sig):
    return np.exp(-(x - mu)**2 / (2 * sig**2)) / ( sig * np.sqrt(2 * np.pi))

def f1b():
    n = 12
    v = np.arange(n)
    e = np.array([0.1]*n) * 10e-0
    v = np.arange(n)**0.75 * 0.2
    e = (np.arange(n)+1)**0.7 * 1e-1
    w = np.arange(n)+1
    print('total:', *Sumb(v,e))
    j = n//3
    a1,e1 = Sumb(v[:j],e[:j])
    a2,e2 = Sumb(v[j:],e[j:])

    print('part1:',a1,e1)
    print('part2:',a2,e2)
    
    vx = np.array([a1,a2])
    ex = np.array([e1,e2])
    af,ef = Sumb(vx,ex)
    print('combi:',af,ef)

    f = plt.figure()
    a = f.add_subplot(111)
    dx = 0.0001
    x = np.arange(-1,v[-1]+2,dx)
    c = ['b']*j + ['g']*(n-j)
    for i in range(n):
        a.plot(x,gaussian(x,v[i],e[i]),color=c[i])
    a.plot(x,gaussian(x,a1,e1),color=c[0],lw=2)
    a.plot(x,gaussian(x,a2,e2),color=c[-1],lw=2)
    a.plot(x,gaussian(x,af,ef),color='r',lw=2)

    plt.draw()

def Sumw(v:'value', e:'error', w:'weight'=None):
    n = len(v)
    assert len(v) == len(e)
    if w is None:
        w = np.array([1.]*len(v))
    else:
        assert len(w) == len(v)
    wx = np.sum(w)
    w = np.array(w) * e**(-2) 
    wt = 1/np.sum(w) 
    a = np.sum(v * w) * wt 
    e2 = wt*wx
    return a, np.sqrt(e2), wx

def f1w():
    n = 12
    v = np.arange(n)
    e = np.array([0.1]*n) * 10e-0
    v = np.arange(n)**0.75 * 0.2
    e = (np.arange(n)+1)**0.7 * 1e-1
    w = np.arange(n)**2+1
    print('total:', *Sumw(v,e,w))
    j = n//3
    a1,e1,w1 = Sumw(v[:j],e[:j],w[:j])
    a2,e2,w2 = Sumw(v[j:],e[j:],w[j:])

    print('part1:',a1,e1,w1)
    print('part2:',a2,e2,w2)
    
    vx = np.array([a1,a2])
    ex = np.array([e1,e2])
    wx = np.array([w1,w2])
    af,ef,wf = Sumw(vx,ex,wx)
    print('combi:',af,ef,wf)

    f = plt.figure()
    a = f.add_subplot(111)
    dx = 0.0001
    x = np.arange(-1,v[-1]+2,dx)
    c = ['b']*j + ['g']*(n-j)
    for i in range(n):
        a.plot(x,gaussian(x,v[i],e[i]/np.sqrt(w[i])),color=c[i])
    a.plot(x,gaussian(x,a1,e1/np.sqrt(w1)),color=c[0],lw=2)
    a.plot(x,gaussian(x,a2,e2/np.sqrt(w2)),color=c[-1],lw=2)
    a.plot(x,gaussian(x,af,ef/np.sqrt(wf)),color='r',lw=2)

    plt.draw()


def lg(x, mu, sig):
    return -(x - mu)**2 / (2 * sig**2) - np.log( sig * np.sqrt(2 * np.pi))

def f2():
    n = 4
    v = np.arange(n)**0.75 * 0.2
    e = (np.arange(n)+1)**0.7 * 1e-1

    n = 12
    v = np.arange(n)
    e = np.array([0.1]*n) * 10e-0

    print(Sumb(v,e))

    f = plt.figure()
    a = f.add_subplot(111)

    dx = 0.0001
    x = np.arange(-1,v[-1]+1,dx)
    y = x.copy()
    y[:] = 0.
    for i in range(n):
        yx = lg(x,v[i],e[i])
        a.plot(x,np.exp(yx),label='{:d}'.format(i))
        y += yx
    y = np.exp(y - np.max(y))
    y /= np.sum(y) * dx    
    a.plot(x,y,label='sum')
    s = np.argsort(y)[::-1]
    ys = np.cumsum(y[s]) * dx
    yi = np.argwhere(ys > 0.682689492137)[0][0]
    print('mean  = {:2f}'.format(x[s[0]]))
    print('sigma = {:2f}'.format(yi*dx/2))
    xy = np.ndarray((yi+2,2))
    i0,i1 = min(s[:yi]), max(s[:yi])
    xy[:yi,0] = x[i0:i1+1]
    xy[:yi,1] = y[i0:i1+1]
    xy[yi:,1] = 0
    xy[yi:,0] = x[[i1,i0]]
    a.add_patch(Polygon(xy,fill=True,color='green',ec='none',alpha=0.25))
    
    leg = plt.legend()

    plt.draw()


def f3():  # mass distribution
    n = 4
    v = np.arange(n)**0.75 * 0.2
    e = (np.arange(n)+1)**0.7 * 1e-1

    n = 12
    v = np.arange(n)**0.75
    e = (np.arange(n)**1+1) * 1e-2

    a0,e0 = Sum(v,e) 
    print(a0,e0)

    f = plt.figure()
    a = f.add_subplot(111)

    dx = 0.0001
    x = np.arange(-1,v[-1]+1,dx)
    y = x.copy()
    y[:] = 0.
    for i in range(n):
        yx = gaussian(x,v[i],e[i])
        a.plot(x,yx,label='{:d}'.format(i))
        y += yx
    y /= np.sum(y) * dx    
    a.plot(x,y,label='sum')
    # s = np.argsort(y)[::-1]
    # ys = np.cumsum(y[s]) * dx
    # yi = np.argwhere(ys > 0.682689492137)[0][0]
    # print('mean  = {:2f}'.format(x[s[0]]))
    # print('sigma = {:2f}'.format(yi*dx/2))

    # xy = np.ndarray((yi+2,2))
    # i0,i1 = min(s[:yi]), max(s[:yi])
    # xy[:yi,0] = x[i0:i1+1]
    # xy[:yi,1] = y[i0:i1+1]
    # xy[yi:,1] = 0
    # xy[yi:,0] = x[[i1,i0]]
    # a.add_patch(Polygon(xy,fill=True,color='green',ec='none',alpha=0.25))
    
    av = np.sum(x*y) / np.sum(y)
    si = np.sqrt((np.sum(x**2*y) / np.sum(y) - av**2))
    
    print('mean  = {:2f}'.format(av))
    print('sigma = {:2f}'.format(si))
    a.plot(x,gaussian(x,a0,e0),lw=1,color='k',label='G')
    a.plot(x,gaussian(x,av,si),lw=2,color='k',label='F')

    ab,eb = Sumb(v,e)
    print(ab,eb)
    a.plot(x,gaussian(x,ab,eb),lw=2,color='k',ls='--',label='B')    

    am,em = SumM(v,e)
    print(am,em)
    a.plot(x,gaussian(x,am,em),lw=2,color='k',ls='-.',label='M')    
    

    leg = plt.legend()
    a.set_xlim(-1,8)
    a.set_ylim(0, 0.35)

    plt.draw()

def f4():
    """
    test - same as Bayes's probability product,
           but we take the ruslt to power 1/n**2
    """
    n = 4
    v = np.arange(n)**0.75 * 0.2
    e = (np.arange(n)+1)**0.7 * 1e-1

    n = 12
    v = np.arange(n)
    e = np.array([0.1]*n) * 10e-0

    print(Sumb(v,e))

    f = plt.figure()
    a = f.add_subplot(111)

    dx = 0.0001
    x = np.arange(-1,v[-1]+1,dx)
    y = x.copy()
    y[:] = 0.
    for i in range(n):
        yx = lg(x,v[i],e[i])
        a.plot(x,np.exp(yx),label='{:d}'.format(i))
        y += yx
    y = np.exp((y - np.max(y))/n**2)
    y /= np.sum(y) * dx    
    a.plot(x,y,label='sum')
    s = np.argsort(y)[::-1]
    ys = np.cumsum(y[s]) * dx
    yi = np.argwhere(ys > 0.682689492137)[0][0]
    print('mean  = {:2f}'.format(x[s[0]]))
    print('sigma = {:2f}'.format(yi*dx/2))
    xy = np.ndarray((yi+2,2))
    i0,i1 = min(s[:yi]), max(s[:yi])
    xy[:yi,0] = x[i0:i1+1]
    xy[:yi,1] = y[i0:i1+1]
    xy[yi:,1] = 0
    xy[yi:,0] = x[[i1,i0]]
    a.add_patch(Polygon(xy,fill=True,color='green',ec='none',alpha=0.25))
    
    leg = plt.legend()
    plt.draw()


