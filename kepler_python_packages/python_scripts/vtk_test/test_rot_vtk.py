
"""
Sphere visualisations
"""

import vtk
from vtk.util.numpy_support import vtk_to_numpy

import time
import numpy as np
from numpy import linalg as LA

from .utils import write_image
from .utils import VtkBase
from .utils import MovieWriter, NoMovieWriter

from .rot import rotate2, Rotator, rotscale, w2axyz
from .rot import w2zyx, w2xyz


class Shells(VtkBase):
    def __init__(self, **kwargs):

        r1 = kwargs.setdefault('r1', 0.9)
        r2 = kwargs.setdefault('r2', 1.1)
        res = kwargs.setdefault('res', 120)

        c1 = np.array([0.,1.,0.5])
        c2 = np.array([0.,0.5,1.0])

        actors = []

        o2 = 1 - 2 * np.random.rand(3)
        o2 /= LA.norm(o2)
        print(o2)

        # o2 = np.array([0.,1.,0.])

        # Create a Sphere 1
        s1 = vtk.vtkSphereSource()
        s1.SetCenter(0.0, 0.0, 0.0)
        s1.SetRadius(r1)
        s1.SetThetaResolution(res)
        s1.SetPhiResolution(res)
        s1.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)

        # Create a mapper and actor
        m1 = vtk.vtkPolyDataMapper()
        m1.SetInputConnection(s1.GetOutputPort())

        a1 = vtk.vtkActor()
        p1 = a1.GetProperty()
        p1.SetColor(c1)
        p1.SetOpacity(0.1)
        # prop.SetAmbient(0.5)
        # prop.SetDiffuse(0.6)
        # prop.SetSpecular(1.0)
        # prop.SetSpecularPower(10.0)
        a1.SetMapper(m1)
        actors.append(a1)

        # Create a Sphere 2
        s2 = vtk.vtkSphereSource()
        s2.SetCenter(0.0, 0.0, 0.0)
        s2.SetRadius(r2)
        s2.SetThetaResolution(res)
        s2.SetPhiResolution(res)
        s2.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)

        # Create a mapper and actor
        m2 = vtk.vtkPolyDataMapper()
        m2.SetInputConnection(s2.GetOutputPort())

        a2 = vtk.vtkActor()
        p2 = a2.GetProperty()
        p2.SetColor(c2)
        p2.SetOpacity(0.1)
        # prop.SetAmbient(0.5)
        # prop.SetDiffuse(0.6)
        # prop.SetSpecular(1.0)
        # prop.SetSpecularPower(10.0)
        a2.SetMapper(m2)
        actors.append(a2)

        # arrow 1

        # w1 = np.asarray([0,1,0.])
        w1 = np.asarray([-1,0.5,-1], dtype = np.float)
        w1 /= LA.norm(w1)

        s3 = vtk.vtkArrowSource()
        s3.SetTipResolution(100)
        s3.SetShaftResolution(100)
        m3 = vtk.vtkPolyDataMapper()
        m3.SetInputConnection(s3.GetOutputPort())

        a3 = vtk.vtkActor()
        p3 = a3.GetProperty()
        p3.SetColor(c1)
        p3.SetOpacity(1.)
        # prop.SetAmbient(0.5)
        # prop.SetDiffuse(0.6)
        # prop.SetSpecular(1.0)
        # prop.SetSpecularPower(10.0)

        # a3.SetScale(np.asarray([1.,1.,1.])*LA.norm(w1))
        # a3.RotateWXYZ(*w2axyz(rotate2(w1), deg = True))
        w = rotate2(w1)
        z, y, x = w2zyx(w, deg = True)
        x, y, z = w2xyz(w, deg = True)
        print(x,y,z)
        a3.RotateZ(z)
        a3.RotateY(y)
        a3.RotateX(x)

        a3.SetPosition(w1 * r1 / LA.norm(w1))

        a3.SetMapper(m3)
        actors.append(a3)

        # arrow 2

        # w2 = np.asarray([0.2,1,0])*0.8
        # w2 = np.asarray([0.2,1,0])
        w2 = np.asarray([-1,-1.1,.8], dtype=np.float)
        w2 /= LA.norm(w2)

        s4 = vtk.vtkArrowSource()
        s4.SetTipResolution(100)
        s4.SetShaftResolution(100)
        m4 = vtk.vtkPolyDataMapper()
        m4.SetInputConnection(s4.GetOutputPort())

        a4 = vtk.vtkActor()
        p4 = a4.GetProperty()
        p4.SetColor(c2)
        p4.SetOpacity(1.)
        # prop.SetAmbient(0.5)
        # prop.SetDiffuse(0.6)
        # prop.SetSpecular(1.0)
        # prop.SetSpecularPower(10.0)


        #a4.SetScale(np.asarray([1.,1.,1.])*LA.norm(w2))

        # a4.RotateZ(90.)

        a4.RotateWXYZ(*w2axyz(rotate2(o2), deg = True))
        # w = rotate2(w2, o2)
        z, y, x = w2zyx(rotate2(w2), deg = True)
        # x, y, z = w2xyz(rotate2(w2), deg = True)
        # print(x,y,z)
        a4.RotateX(x)
        a4.RotateY(y)
        a4.RotateZ(z)

        a4.RotateWXYZ(*w2axyz(rotate2(o2), deg = True))
        a4.RotateWXYZ(*w2axyz(rotate2(w2, o2), deg = True))

        # a4.RotateWXYZ(*w2axyz(rotate2(w2), deg = True))


        #print(a4.GetOrientationWXYZ())
        #print(rotate2(w2))


        # a4.SetOrientation(w2xyz(rotate2(w2), deg = True))
        a4.SetPosition(w2 * r2 / LA.norm(w2))

        a4.SetMapper(m4)
        actors.append(a4)

        # Create a Sphere as pointer target
        s = vtk.vtkSphereSource()
        s.SetCenter(0.0, 0.0, 0.0)
        s.SetRadius(.1)
        s.SetThetaResolution(res)
        s.SetPhiResolution(res)
        s.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)

        # Create a mapper and actor
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(s.GetOutputPort())

        # S2
        a = vtk.vtkActor()
        p = a.GetProperty()
        p.SetColor([1.,1.,0])
        p.SetOpacity(1.)
        a.SetMapper(m)

        w = rotate2(w2, o2)
        v = Rotator(w)(o2 * 2.2)
        a.SetPosition(v)
        actors.append(a)

        # S1
        a = vtk.vtkActor()
        p = a.GetProperty()
        p.SetColor([1.,0.,0])
        p.SetOpacity(1.)
        a.SetMapper(m)

        w = rotate2(w1)
        v = Rotator(w)(np.array([2.,0.,0.]))
        a.SetPosition(v)
        actors.append(a)

        # Create a renderer, render window, and interactor
        renderer = vtk.vtkRenderer()
        renderer.BackingStoreOn()
        # renderer.UseShadowsOn()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetSize(800, 600)

        renderWindow.SetWindowName("Shells")
        renderWindow.AddRenderer(renderer)
        renderWindow.SetAlphaBitPlanes(1)

        # nvida pixel depth
        # renderWindow.SetAlphaBitPlanes(1)
        # renderWindow.SetMultiSamples(0)

        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        # Add the actor to the scene
        for a in actors[::-1]:
            renderer.AddActor(a)

        # set background
        renderer.SetBackground([0.8, 0.8, 0.8])
        renderer.SetBackground2([0.2, 0.2, 0.2])
        renderer.GradientBackgroundOn()

        # camera
        camera = vtk.vtkCamera()
        renderer.SetActiveCamera(camera)
        # renderer.ResetCamera()
        # camera.SetPosition(0., 1., 7.)
        # camera.SetFocalPoint(0., 0.5, 0.)
        camera.SetPosition(0., 0., 7.)
        camera.SetFocalPoint(0., 0.0, 0.0)

        # light
        light = vtk.vtkLight()
        light.SetFocalPoint(0., 0., 0.)
        light.SetPosition(3., 5., 1.)
        renderer.AddLight(light)

        # Render
        renderWindowInteractor.Initialize()
        renderWindow.Render()

        # save stuff
        self.renderer = renderer
        self.renderWindow = renderWindow
        self.renderWindowInteractor = renderWindowInteractor
        self.camera = camera

        self.w1 = w1
        self.w2 = w2

        self.r1 = r1
        self.r2 = r2

        self.c1 = c1
        self.c2 = c2


        self.renderWindow.Render()


    def animate(self, movie = None, alpha = False):

        camera = self.camera
        renderWindow = self.renderWindow
        renderer = self.renderer

        if movie is not None:
            frames = []

        if alpha == True:
            renderer.SetBackground([1.,1.,1.])
            renderer.GradientBackgroundOff()

        # renderer.ResetCamera()
        # camera.SetViewUp(0,1,0)
        camera.SetFocalPoint(0.,0.,0.)
        # a movie
        nstep = 300
        for i in np.arange(0, nstep):
            i = i / nstep * 2 * np.pi
            camera.SetPosition(0, 8*np.sin(i), 8*np.cos(i))
            camera.SetViewUp(0, np.cos(i), -np.sin(i))
            renderWindow.Render()
            time.sleep(.033)
            if movie is None:
                # time.sleep(.033)
                pass
            else:
                frames.append(self.get_image_array(alpha=True))

        if movie is not None:
            write_movie(frames, movie)
