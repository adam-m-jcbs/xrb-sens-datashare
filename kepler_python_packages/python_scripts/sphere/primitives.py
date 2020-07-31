

import numpy as np
from numpy import linalg as LA

import vtk

from rotation import rotate2, w2axyz

def tube(v, r, c, nTv = 50, alpha = None, capping = True):

    nV = len(v)

    if np.shape(r) == ():
        r = np.tile(r, nV)
    c = np.array(c)
    if len(c.shape) == 1:
        c = np.tile(c, (nV, 1))
    nc = c.shape[1]
    if alpha is None and nc == 3:
        alpha = 1.
    if alpha is not None:
        if np.shape(alpha) == ():
            alpha = np.tile(alpha, nV)
        if nc == 4:
            c[:, 3] *= alpha
        else:
            c = np.hstack((c, alpha[:, np.newaxis]))
            nc = 4

    # Create points and cells for line
    points = vtk.vtkPoints()

    for i in range(nV):
        points.InsertPoint(i, *v[i]);

    lines = vtk.vtkCellArray()
    lines.InsertNextCell(nV)
    for i in range(nV):
        lines.InsertCellPoint(i)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(lines)

    # Varying tube radius using sine-function
    tubeRadius = vtk.vtkDoubleArray()
    tubeRadius.SetName("TubeRadius")
    tubeRadius.SetNumberOfTuples(nV)
    for i in range (nV):
        tubeRadius.SetTuple1(
            i,
            r[i],
            )
    polyData.GetPointData().AddArray(tubeRadius)
    polyData.GetPointData().SetActiveScalars("TubeRadius")

    colors = vtk.vtkUnsignedCharArray()
    colors.SetName("Colors")
    colors.SetNumberOfComponents(nc)
    colors.SetNumberOfTuples(nV)
    for i in range(nV):
        colors.InsertTuple4(i, *np.round(c[i] * 255))
    polyData.GetPointData().AddArray(colors)

    tube = vtk.vtkTubeFilter()
    tube.SetInputData(polyData)
    tube.SetCapping(capping)

    tube.SetNumberOfSides(nTv)
    tube.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tube.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())
    mapper.ScalarVisibilityOn();
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("Colors")

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

def ca2ca(c, alpha):
    c = np.asarray(c)
    if alpha is None:
        if c.shape[0] == 4:
            alpha = c[3]
            c = c[:3]
        else:
            alpha = 1.
    else:
        if c.shape[0] == 4:
            alpha *= c[3]
            c = c[:3]
    return c, alpha

def cone(radius, height, scale, pos, u, c, alpha = None, res = 100, capping = True):
    c, alpha = ca2ca(c, alpha)
    s = vtk.vtkConeSource()
    s.SetResolution(res)
    s.SetRadius(float(radius))
    s.SetHeight(float(height))
    s.SetCenter(np.asarray([height * 0.5, 0., 0.]))
    s.SetCapping(capping)
    # s.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(s.GetOutputPort())
    a = vtk.vtkActor()
    p = a.GetProperty()
    p.SetColor(c)
    p.SetOpacity(alpha)
    a.SetMapper(m)
    a.SetScale(np.asarray([1.,1.,1.]) * scale)
    a.RotateWXYZ(*w2axyz(rotate2(u), deg = True))
    a.SetPosition(pos)
    return a

def cylinder(pos, radius, height, u, c,
             alpha = None,
             capping = True,
             res = 100,
             ):
    c, alpha = ca2ca(c, alpha)
    s = vtk.vtkCylinderSource()
    s.SetCenter(0.0, 0.0, 0.0)
    s.SetRadius(radius)
    s.SetHeight(height)
    s.SetResolution(res)
    s.SetCapping(capping)
    s.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)

    # Create a mapper and actor
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(s.GetOutputPort())

    a = vtk.vtkActor()
    p = a.GetProperty()
    p.SetColor(c)
    p.SetOpacity(alpha)
    # prop.SetAmbient(0.5)
    # prop.SetDiffuse(0.6)
    # prop.SetSpecular(1.0)
    # prop.SetSpecularPower(10.0)
    a.SetMapper(m)

    a.RotateWXYZ(*w2axyz(rotate2(u, [0., 1., 0.]), deg = True))
    a.SetPosition(pos)

    return a

def sphere(pos, radius, c,
           alpha = None,
           res = 100,
             ):
    c, alpha = ca2ca(c, alpha)

    s = vtk.vtkSphereSource()
    s.SetCenter([0., 0., 0.])
    s.SetRadius(radius)
    s.SetThetaResolution(res)
    s.SetPhiResolution(res)
    s.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)

    # Create a mapper and actor
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(s.GetOutputPort())

    a = vtk.vtkActor()
    p = a.GetProperty()
    p.SetColor(c)
    p.SetOpacity(alpha)
    # prop.SetAmbient(0.5)
    # prop.SetDiffuse(0.6)
    # prop.SetSpecular(1.0)
    # prop.SetSpecularPower(10.0)
    a.SetMapper(m)
    a.SetPosition(pos)
    return a

def rotation_ellipsoid(pos, axis, radius, c,
           alpha = None,
           res = 100,
             ):
    '''
    axis:
       location and radius in that direction

    radius:
       radius in perpendicular direction
    '''

    c, alpha = ca2ca(c, alpha)

    axis = np.asarray(axis)
    an = LA.norm(axis)

    s = vtk.vtkSphereSource()
    s.SetCenter([0., 0., 0.])
    s.SetRadius(radius)
    s.SetThetaResolution(res)
    s.SetPhiResolution(res)
    s.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)

    # Create a mapper and actor
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(s.GetOutputPort())

    a = vtk.vtkActor()
    p = a.GetProperty()
    p.SetColor(c)
    p.SetOpacity(alpha)
    # prop.SetAmbient(0.5)
    # prop.SetDiffuse(0.6)
    # prop.SetSpecular(1.0)
    # prop.SetSpecularPower(10.0)
    a.SetMapper(m)
    a.SetScale(np.asarray([an / radius, 1., 1.]))
    a.RotateWXYZ(*w2axyz(rotate2(axis), deg = True))

    a.SetPosition(pos)
    return a

def arrow(pos, u, size, c,
          alpha = None,
          tipradius = 0.1,
          shaftradius = 0.03,
          tiplength = 0.35,
          res = None,
          tipresolution = 120,
          shaftresolution = 60,
          ):
    c, alpha = ca2ca(c, alpha)

    if res is not None:
        tipresolution = 2 * res
        shaftesolution = res

    s = vtk.vtkArrowSource()
    s.SetTipLength(tiplength)
    s.SetTipRadius(tipradius)
    s.SetShaftResolution(shaftresolution)
    s.SetTipResolution(tipresolution)
    s.SetShaftRadius(shaftradius)
    # s.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(s.GetOutputPort())

    a = vtk.vtkActor()
    p = a.GetProperty()
    p.SetColor(c)
    p.SetOpacity(alpha)
    # prop.SetAmbient(0.5)
    # prop.SetDiffuse(0.6)
    # prop.SetSpecular(1.0)
    # prop.SetSpecularPower(10.0)

    a.SetScale(np.asarray([1., 1., 1.]) * size)
    a.RotateWXYZ(*w2axyz(rotate2(u), deg = True))
    # w = rotate2(w2)
    # x, y, z = w2xyz(u, deg = True)
    # print(x,y,z)
    # a4.RotateZ(z)
    # a4.RotateY(y)
    # a4.RotateX(x)
    # a4.SetOrientation(w2xyz(rotate2(w2), deg = True))
    a.SetPosition(pos)

    a.SetMapper(m)
    return a
