
"""
Sphere visualisations
"""

import vtk
from vtk.util.numpy_support import vtk_to_numpy

import time
import numpy as np
from numpy import linalg as LA

import PIL

from .image import write_image
from .movie import write_movie
from .vtkbase import VtkBase

from .rot import w2zyx, rotate2, w2xyz

class Spiral(VtkBase):
    def __init__(self, **kwargs):
        nV = 2560 # No. of vertices
        nCyc = 5 # No. of spiral cycles
        rT1 = 0.1 # Start tube radius
        rT2 = 0.5 # Eend tube radius
        rS = 2 # Spiral radius
        h = 10 # Height
        nTv = 80 #No. of surface elements for each tube vertex

        # Create points and cells for the spiral
        points = vtk.vtkPoints()
        for i in range(nV):
            # Spiral coordinates
            vX = rS * np.cos(2 * np.pi * nCyc * i / (nV - 1))
            vY = rS * np.sin(2 * np.pi * nCyc * i / (nV - 1))
            vZ = h * i / nV
            points.InsertPoint(i, vX, vY, vZ);
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
                rT1 + (rT2 - rT1) * np.sin(np.pi * i / (nV - 1)),
                )

        polyData.GetPointData().AddArray(tubeRadius)
        polyData.GetPointData().SetActiveScalars("TubeRadius")

        # RBG array (could add Alpha channel too I guess...)
        # Varying from blue to red
        colors = vtk.vtkUnsignedCharArray()
        colors.SetName("Colors")
        colors.SetNumberOfComponents(4)
        colors.SetNumberOfTuples(nV)
        for i in range(nV):
            colors.InsertTuple4(
                i,
                int(255 * i/ (nV - 1)),
                0,
                int(255 * (nV - 1 - i) / (nV - 1)),
                255 #int(255 * (0.2 + 0.8 * np.cos( i / (nV - 1)))),
                )

        polyData.GetPointData().AddArray(colors)

        tube = vtk.vtkTubeFilter()
        tube.SetInputData(polyData)
        tube.CappingOn()
        # tube.SetOnRatio(2)

        tube.SetNumberOfSides(nTv)
        tube.SetVaryRadiusToVaryRadiusByAbsoluteScalar()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        mapper.ScalarVisibilityOn();
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("Colors")

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(.2, .3, .4)

        # Make an oblique view
        renderer.GetActiveCamera().Azimuth(30)
        renderer.GetActiveCamera().Elevation(30)
        renderer.ResetCamera()
        # renderer.UseShadowsOn()

        renWin = vtk.vtkRenderWindow()
        renWin.SetAlphaBitPlanes(1)
        iren = vtk.vtkRenderWindowInteractor()

        iren.SetRenderWindow(renWin)
        renWin.AddRenderer(renderer)
        renWin.SetSize(500, 500)
        renWin.Render()

        style = vtk.vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(style)

        iren.Start()
