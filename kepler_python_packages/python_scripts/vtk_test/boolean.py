
"""
Boolean visualisations
"""

# https://vtk.org/doc/nightly/html/c2_vtk_t_9.html#c2_vtk_t_vtkIntersectionPolyDataFilter

import vtk
from vtk.util.numpy_support import vtk_to_numpy

import time
import numpy as np
from numpy import linalg as LA

import PIL

from .utils import write_PIL_image
from .utils import write_PIL_movie
from .utils import VtkBase

from .rot import w2zyx, rotate2, w2xyz

class Boolean(VtkBase):
    def get_boolean_operation_actor(self, x, operation, **kwargs):
        centerSeparation = 0.15

        res = 120

        sphere1 = vtk.vtkSphereSource()
        sphere1.SetCenter(-centerSeparation + x, 0.0, 0.0)
        sphere1.SetThetaResolution(res)
        sphere1.SetPhiResolution(res)


        sphere2 = vtk.vtkSphereSource()
        sphere2.SetCenter(  centerSeparation + x, 0.0, 0.0)
        sphere2.SetThetaResolution(res)
        sphere2.SetPhiResolution(res)

        intersection = vtk.vtkIntersectionPolyDataFilter()
        intersection.SetInputConnection( 0, sphere1.GetOutputPort() )
        intersection.SetInputConnection( 1, sphere2.GetOutputPort() )

        distance = vtk.vtkDistancePolyDataFilter()
        distance.SetInputConnection( 0, intersection.GetOutputPort( 1 ) )
        distance.SetInputConnection( 1, intersection.GetOutputPort( 2 ) )

        thresh1 = vtk.vtkThreshold()
        thresh1.AllScalarsOn()
        thresh1.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Distance" )
        thresh1.SetInputConnection( distance.GetOutputPort( 0 ) )

        thresh2 = vtk.vtkThreshold()
        thresh2.AllScalarsOn()
        thresh2.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Distance" )
        thresh2.SetInputConnection( distance.GetOutputPort( 1 ) )

        if operation == vtk.vtkBooleanOperationPolyDataFilter.VTK_UNION:
            thresh1.ThresholdByUpper( 0.0 )
            thresh2.ThresholdByUpper( 0.0 )
        elif operation == vtk.vtkBooleanOperationPolyDataFilter.VTK_INTERSECTION:
            thresh1.ThresholdByLower( 0.0 )
            thresh2.ThresholdByLower( 0.0 )
        else:
            # Difference
            thresh1.ThresholdByUpper( 0.0 )
            thresh2.ThresholdByLower( 0.0 )

        surface1 = vtk.vtkDataSetSurfaceFilter()
        surface1.SetInputConnection( thresh1.GetOutputPort() )

        surface2 = vtk.vtkDataSetSurfaceFilter()
        surface2.SetInputConnection( thresh2.GetOutputPort() )

        reverseSense = vtk.vtkReverseSense()
        reverseSense.SetInputConnection( surface2.GetOutputPort() )
        if operation == 2:
            # difference
            reverseSense.ReverseCellsOn()
            reverseSense.ReverseNormalsOn()

        appender = vtk.vtkAppendPolyData()
        appender.SetInputConnection( surface1.GetOutputPort() )
        if operation == 2:
            appender.AddInputConnection( reverseSense.GetOutputPort() )
        else:
            appender.AddInputConnection( surface2.GetOutputPort() )

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection( appender.GetOutputPort() )
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper( mapper )

        return actor

    def __init__(self, **kwargs):

        renderer = vtk.vtkRenderer()

        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer( renderer )

        renWinInteractor = vtk.vtkRenderWindowInteractor()
        renWinInteractor.SetRenderWindow( renWin )

        unionActor = self.get_boolean_operation_actor( -2.0, vtk.vtkBooleanOperationPolyDataFilter.VTK_UNION )
        renderer.AddActor( unionActor )
        del unionActor

        intersectionActor = self.get_boolean_operation_actor(  0.0, vtk.vtkBooleanOperationPolyDataFilter.VTK_INTERSECTION )
        renderer.AddActor( intersectionActor )
        del intersectionActor

        differenceActor = self.get_boolean_operation_actor(  2.0, vtk.vtkBooleanOperationPolyDataFilter.VTK_DIFFERENCE )
        renderer.AddActor( differenceActor )
        del differenceActor

        renWin.Render()
        renWinInteractor.Start()
