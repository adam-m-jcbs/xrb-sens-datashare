"""
Python module to plot 3D SN.

(under construction)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from mpl_toolkits.mplot3d.art3d import pathpatch_2d_to_3d, Patch3D, Poly3DCollection

def tetraeder(radius = 1.0):
    return np.array([[-1,-1,-1],[+1,+1,-1],[-1,+1,+1],[+1,-1,+1]], 
                    dtype=np.float64)/np.sqrt(3.)*radius

def tess(vertices, triangles):
    nv = vertices.shape[0]
    nt = triangles.shape[0]
    vert_new = np.ndarray((nv+nt,3), dtype=np.float64)
    tria_new = np.ndarray((3*nt,3), dtype=np.uint64)
    vert_new[0:nv] = vertices
    for i in range(nt):
        x = vertices[triangles[i,0]] + vertices[triangles[i,1]] + vertices[triangles[i,2]]
        x /= np.sqrt(np.sum(x**2))
        j = nv+i
        vert_new[j] = x
        tria_new[3*i+0] = [triangles[i,0],triangles[i,1],j]
        tria_new[3*i+1] = [triangles[i,1],triangles[i,2],j]
        tria_new[3*i+2] = [triangles[i,2],triangles[i,0],j]
    return vert_new, tria_new

def triquad(vertices, triangles):
    nv = vertices.shape[0]
    nt = triangles.shape[0]

    nv_new = (nt*5)//2
    nt_new = 4*nt
    vert_new = np.ndarray((nv_new,3), dtype=np.float64)
    tria_new = np.ndarray((nt_new,3), dtype=np.uint64)

    vert_new[0:nv] = vertices
       
    d = dict()
    for k,t in enumerate(triangles):
        for i0 in range(3):
            i1 = (i0 + 1) % 3
            t0 = t[i0]
            t1 = t[i1]
            m = min(t0, t1)
            M = max(t0, t1)
            c = (m, M) 
            try:
                tn = d[c]
            except:
                d[c] = tn = len(d) + nt
                vert_new[tn] = vertices[t0] + vertices[t1]
            tria_new[4*k+i0 , 0] = t0
            tria_new[4*k+i0 , 1] = tn
            tria_new[4*k+i1 , 2] = tn
            tria_new[4*k+ 3 ,i0] = tn

    i = slice(nv,nv_new,1)
    vert_new[i] /= np.sqrt(np.sum(vert_new[i]**2,1))[:,np.newaxis]
    return vert_new, tria_new

def split(vertices, triangles):
    d = dict()
    for k,t in enumerate(triangles):
        for i0 in range(3):
            i1 = (i0 + 1) % 3
            t0 = t[i0]
            t1 = t[i1]
            m = min(t0, t1)
            M = max(t0, t1)
            c = (m, M)
            v = [t,i0,i1]
            try:
                d[c][2] = v
            except:        
                dist = np.sum((vertices[t0] - vertices[t1])**2)
                d[c] = [dist, v, None]
    edges = list(d.values())
    i = np.argsort(np.array([e[0] for e in edges]))[::-1]
    
def sphere(n = 200, radius = 10):

    vertices = tetraeder()
    triangles = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
    
    vertices, triangles = triquad(vertices, triangles)
    # vertices, triangles = triquad(vertices, triangles)
    # vertices, triangles = triquad(vertices, triangles)
    # vertices, triangles = triquad(vertices, triangles)
    split(vertices, triangles)


    dx = dy = dz = 0.5
    d = np.array([dx,dy,dz])

    x, y, z = vertices.transpose()*radius - 0.5*d[:,np.newaxis]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d',aspect='equal')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    shape = []
    for t in triangles:
        shape += [np.array([vertices[t[0]],vertices[t[1]],vertices[t[2]]])*radius]

    p = Poly3DCollection(shape, color = 'r', edgecolor = 'k')
    ax.add_collection3d(p)


    ax.bar3d(x, y, z, dx, dy, dz, 
             color=np.array([0.,.8,.2]).transpose(), 
             edgecolor='none',
             zsort='average', 
             alpha=1.)

    plt.show()   




def testX(n = 200):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d',aspect='equal')
    x, y, z = np.random.rand(3, n) * 10
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    dx = dy = dz = 0.5
    
    # ax.bar3d(x, y, z, dx, dy, dz, 
    #          color=np.array([0*x,x/50,(1-x/10)/5]).transpose()+0.8, 
    #          edgecolor='none',
    #          zsort='average', 
    #          alpha=1)
    ax.bar3d(x, y, z, dx, dy, dz, 
             color=np.array([0.,.8,.2]).transpose(), 
             edgecolor='none',
             zsort='average', 
             alpha=.1)

    plt.show()   

