"""
Some ideas on how to combin CAIRO graphics with Movie framework
"""

import numpy as np
import cairo

from PIL import Image
# from io import BytesIO

from .frames import MocieCanvasBase

class CairoBase(MovieCanvasBase):
    def __init__(self,
                 size = (800, 600),
                 mode = cairo.FORMAT_ARGB32,
                 surface = None,
                 color = None,
                 ):
        self.mode = mode
        self.size = size
        self.color = color
        if surface is None:
            self.surface = cairo.ImageSurface(self.mode, *self.size)
            if self.color is not None:
                self.set_color()
        else:
            self.surface = surface
        self.ctx = cairo.Context(self.surface)
        self.ctx.scale(*self.size)

    def _set_color(self, color = None):
        if color is None:
            color = (0,0,0,0)
        arr = self._get_arr()
        arr[:,:,:] = color[(2,1,0,3)]
        self.surface.mark_dirty()

    def _get_arr(self):
        """
        return array with direct access to buffer

        note that chanle oreder is reveresed: 2, 1, 0, 3
        """
        self.surface.flush()
        shape = self.get_buf_shape()
        mem = self.surface.get_data()
        arr = np.ndarray(shape, dtype = np.uint8, buffer = mem)
        return arr

    def get_array(self):
        """
        INFO:
        we resort alpha and scale colors to reproduce png output
        """
        self._get_arr()
        arr = arr[:, :, [2, 1, 0, 3]]
        arr[:, :, :3] = arr[:, :, :3] / arr[:, :, 3:] * 255
        return arr

    def get_frame_size(self):
        return (self.surface.get_height(), self.surface.get_width())

    @classmethod
    def from_arr(cls, arr):
        # do we need to resort alpha?
        arr = arr[:, :, [2, 1, 0, 3]].copy()
        arr[:, :, :3] = (arr[:, :, :3] / 255) * arr[:, :, 3:]
        size = arr.shape[:2][::-1]
        data = np.ndarray(size, dtype=np.uint32, buffer=arr.data)
        mode = cairo.FORMAT_ARGB32
        surface = cairo.ImageSurface.create_for_data(data, mode, *size)
        return cls(
            size = size,
            mode = mode,
            surface = surface,
            )

    def write_image(self, filename = 'test.png'):
        self.surface.write_to_png(filename)

    def close(self):
        self.surface.finish()

    def clear(self):
        self._set_color(self.color)


class Demo(CairoBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run()

    def run(self):
        ctx = self.ctx
        pat = cairo.LinearGradient(0.0, 0.0, 0.0, 1.0)
        pat.add_color_stop_rgba(1, 0.7, 0, 0, 0.5)  # First stop, 50% opacity
        pat.add_color_stop_rgba(0, 0.9, 0.7, 0.2, 1)  # Last stop, 100% opacity

        ctx.rectangle(0, 0, 1, 1)  # Rectangle(x0, y0, x1, y1)
        ctx.set_source(pat)
        ctx.fill()

        ctx.translate(0.1, 0.1)  # Changing the current transformation matrix

        ctx.move_to(0, 0)
        # Arc(cx, cy, radius, start_angle, stop_angle)
        ctx.arc(0.2, 0.1, 0.1, -np.pi / 2, 0)
        ctx.line_to(0.5, 0.1)  # Line to (x,y)
        # Curve(x1, y1, x2, y2, x3, y3)
        ctx.curve_to(0.5, 0.2, 0.5, 0.4, 0.2, 0.8)
        ctx.close_path()

        ctx.set_source_rgb(0.3, 0.5, 1)  # Solid color
        ctx.set_line_width(0.02)
        ctx.stroke()

        self.surface.write_to_png("example.png")
