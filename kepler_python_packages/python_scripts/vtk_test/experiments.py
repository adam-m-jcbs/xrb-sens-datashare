#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is just a collection of ideas, examples, and methods.
"""

import vtk
from vtk.util.numpy_support import vtk_to_numpy

import time
import numpy as np
import sys

import PIL


def write_PIL_image(arr, basename = 'xxx', format = 'webp'):
    # write to file
    img =  PIL.Image.fromarray(arr)
    filename = '.'.join((basename, format))
    img.save(
        filename,
        format=format,
        lossless=True,
        quality = 100,
        )


def write_PIL_movie(
        frames,
        basename = 'xxx',
        format = 'webp',
        delay = 17,
        loop = 0,
        ):
    # write to file
    frames = [PIL.Image.fromarray(a) for a in frames]
    filename = '.'.join((basename, format))
    frames[0].save(
        filename,
        format = format,
        lossless = True,
        quality = 100,
        save_all = True,
        duration = delay,
        loop = loop,
        append_images = frames[1:],
        minimize_size = True,
        )


def make_transparent_background(arr):
    # make transpaent background color
    A = np.empty(arr.shape[:-1] + (4,), dtype=arr.dtype)
    A[...,:3] = arr
    A[...,3] = 255
    t = arr[0,0]
    ii = ((arr[...,0]==t[0]) & (arr[...,1]==t[1]) & (arr[...,2]==t[2]))
    A[ii,3]=0
    return A

class Cylinder(object):
    def __init__(self):

        colors = vtk.vtkNamedColors()

        # Create a sphere
        cylinderSource = vtk.vtkCylinderSource()
        cylinderSource.SetCenter(0.0, 0.0, 0.0)
        cylinderSource.SetRadius(5.0)
        cylinderSource.SetHeight(7.0)
        cylinderSource.SetResolution(100)
        cylinderSource.CappingOff()

        # Create a mapper and actor
        #mapper = vtk.vtkPolyDataMapper()
        #mapper.SetInputConnection(cylinderSource.GetOutputPort())


        doDepthsort = True
        scalarVisibility = True

        camera = vtk.vtkCamera()
        camera.SetPosition(0,0,20)

        appendData = vtk.vtkAppendPolyData()
        appendData.AddInputConnection(cylinderSource.GetOutputPort())
        depthSort = vtk.vtkDepthSortPolyData()
        depthSort.SetInputConnection(appendData.GetOutputPort())
        depthSort.SetDirectionToBackToFront()
        depthSort.SetVector(1, 1, 1)
        depthSort.SetCamera(camera)
        depthSort.SortScalarsOn()
        depthSort.Update()
        mapper = vtk.vtkPolyDataMapper()
        if doDepthsort:
            mapper.SetInputConnection(depthSort.GetOutputPort())
        else:
            mapper.SetInputConnection(appendData.GetOutputPort())
        mapper.SetScalarVisibility(scalarVisibility);
        if scalarVisibility:
            mapper.SetScalarRange(0, depthSort.GetOutput().GetNumberOfCells())

        actor = vtk.vtkActor()
        prop = actor.GetProperty()
        prop.SetColor(colors.GetColor3d("Cornsilk"))
        prop.SetOpacity(0.5)
        prop.SetAmbient(0.5)
        prop.SetDiffuse(0.6)
        prop.SetSpecular(1.0)
        prop.SetSpecularPower(10.0)

        #prop.SetAmbientColor(.1,.1,.1)
        #prop.SetDiffuseColor(.1,.2,.4)
        #prop.SetSpecularColor(1,1,1)

        bp = vtk.vtkProperty()

        bp.SetOpacity(0.25)
        bp.SetDiffuse(0.6)
        bp.SetAmbient(0.5)
        bp.SetAmbientColor(0.8000, 0.2, 0.2)
        bp.SetColor(1.,0.,0.)
        bp.SetDiffuse(0.6)
        bp.SetSpecular(1.0)
        bp.SetSpecularPower(3.0)

        actor.SetBackfaceProperty(bp)

        actor.SetMapper(mapper)

        depthSort.SetProp3D(actor)

        # Create a renderer, render window, and interactor
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()

        renderer.SetActiveCamera(camera)
        renderWindow.SetSize(800,600)
        renderer.ResetCamera()

        renderWindow.SetWindowName("Cylinder")
        renderWindow.AddRenderer(renderer)

        # nvida pixel depth
        # renderWindow.SetAlphaBitPlanes(1)
        # renderWindow.SetMultiSamples(0)

        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        # Add the actor to the scene
        renderer.AddActor(actor)
        renderer.SetBackground(colors.GetColor3d("DarkGreen"))

        # Render
        renderWindowInteractor.Initialize()
        renderWindow.Render()

        self.renderer = renderer
        self.renderWindow = renderWindow
        self.renderWindowInteractor = renderWindowInteractor

        self.camera = camera


    def interact(self):
        self.renderWindowInteractor.Start()

    def animate(self, movie = None):

        camera = self.camera
        renderWindow = self.renderWindow
        renderer = self.renderer

        if movie is not None:
            frames = []

        renderer.ResetCamera()
        camera.SetViewUp(0,1,0)
        # a movie
        nstep = 300
        for i in np.arange(0, nstep):
            i = i / nstep*np.pi
            camera.SetPosition(30*np.cos(i),30*np.sin(i),12*np.sin(2*i))
            camera.SetViewUp(-np.sin(i), np.cos(i),0)
            renderWindow.Render()
            if movie is None:
                time.sleep(.033)
            else:
                frames.append(self.get_image_array())

        if movie is not None:
            write_PIL_movie(frames, movie)

    def get_image_array(self):
        renderWindow = self.renderWindow

        # get image data
        vtk_win_im = vtk.vtkWindowToImageFilter()
        vtk_win_im.SetInput(renderWindow)
        vtk_win_im.Update()
        vtk_image = vtk_win_im.GetOutput()
        width, height, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        components = vtk_array.GetNumberOfComponents()
        arr = vtk_to_numpy(vtk_array).reshape(height, width, components)
        return arr[:,:,:]

    def write_image(self, basename = 'xxx', transparent=False):
        a = self.get_image_array()
        if transparent:
            a = make_transparent_background(a)
        write_PIL_image(a, basename)


if __name__ == '__main__':
    c = Cylinder()
    c.interact()
