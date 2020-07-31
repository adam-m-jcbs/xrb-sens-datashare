"""
VTK bases
"""

import numpy as np
import scipy.special
import scipy.misc
import itertools

import vtk
from vtk.util.numpy_support import vtk_to_numpy

from PIL import Image

from .image import make_transparent_background
from .image import write_image
from .frames import MovieCanvasBase

class VtkBase(MovieCanvasBase):
    # currently much of the setup is done in the Scene class of the
    # 'sphere' project

    def __init__(self, size):
        self.size = size

    def get_frame_size(self):
        return self.winsize

    # add from_arr method to set background image

    def interact(self):
        """
        set up basic interaction model
        """
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.renderWindowInteractor.SetInteractorStyle(style)
        self.renderWindowInteractor.Start()

        # TODO - add more interaction utilities, keyboard control, etc.

    def get_array(
            self,
            transparent = False,
            magnification = None,
            antialias = None,
            ):
        if transparent:
            bg = self.renderer.GetBackground()
            gb = self.renderer.GetGradientBackground()
            self.renderer.SetBackground([1, 1, 1])
            self.renderer.GradientBackgroundOff()
            self.renderWindow.Render()
        frame = self.get_buffer(
            alpha = True,
            magnification = magnification,
            antialias = antialias,
            )
        if transparent:
            frame = make_transparent_background(frame)
            self.renderer.SetBackground(bg)
            self.renderer.SetGradientBackground(gb)
            self.renderWindow.Render()
        return frame

    def get_buffer(
            self,
            alpha = False,
            magnification = None,
            antialias = None,
            aamode = 'gamma',
            aagamma = 2.5,
            aaweights = 'lanczos',
            aafilter = 'LANCZOS',
            ):
        renderWindow = self.renderWindow

        # get image data
        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(renderWindow)
        if alpha:
            windowToImageFilter.SetInputBufferTypeToRGBA()
        else:
            windowToImageFilter.SetInputBufferTypeToRGB()
        if magnification is not None:
            scale = magnification
        else:
            scale = 1
        if antialias == True:
            antialias = 2
        if antialias in range(2, 11):
            scale *= antialias
        elif antialias in (None, False):
            pass
        else:
            raise Exception(f'Unknown antialias size "{antialias}".')
        if scale is not 1:
            windowToImageFilter.SetScale(scale)
        windowToImageFilter.ReadFrontBufferOff()
        windowToImageFilter.Update()
        vtk_image = windowToImageFilter.GetOutput()
        width, height, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        components = vtk_array.GetNumberOfComponents()
        arr = vtk_to_numpy(vtk_array).reshape(height, width, components)
        if antialias in (2,3,4):
            n = antialias
            indices = list(itertools.product(range(n), repeat=2))
            if aaweights in ('equal', None):
                weights = np.tile(np.float(n)**(-2), (n,n,))
            elif aaweights in ('binomial',):
                # binomial
                lin = scipy.special.binom(n - 1,np.arange(n)) * 0.5**(n-1)
            elif aaweights in('lanczos', 'Lanczos', 'LANCZOS',):
                # Lanczos kernel
                x = (np.arange(n) - (n-1)/2) / ((n-1)/2) * (n-1)/n
                lin = np.sinc(x)
                lin /= np.sum(lin)
                weights = np.tensordot(lin, lin, axes=0)
            else:
                raise Exception(f'unknown weights {aaweights}')
            if aamode == 'gamma':
                if np.shape(aagamma) == ():
                    aagamma = np.array([aagamma] * arr.shape[-1])
                arr = np.asarray(np.round(np.sum(
                    [weights[i,j] * arr[i::n, j::n, :]**aagamma[...,:]
                     for i,j in indices],
                    axis = 0)**(1/aagamma[..., :])), dtype = np.uint8)
            elif aamode == 'PIL':
                img_filter = getattr(Image, aafilter)
                arr = np.array(Image.fromarray(arr).resize(
                    np.array(arr.shape[1::-1]) // antialias,
                    resample=img_filter))
            elif aamode in (None, 'linear',):
                arr = np.asarray(np.round(np.sum(
                    [weights[i,j] * arr[i::n, j::n, :]
                     for i,j in indices],
                    axis = 0)), dtype = np.uint8)
            else:
                raise Exception(f'Unknown mode {aamode}')
        return np.ascontiguousarray(arr[::-1,:,:])

    def write_image(self, *args, filename = 'test.png', **kwargs):
        frame = self.get_array(*args, **kwargs)
        write_image(frame, filename)

    def get_image(self, *args, **kwargs):
        frame = self.get_array(*args, **kwargs)
        return Image.fromarray(frame)

    def close(self):
        # this needs to be updated properly
        pass
