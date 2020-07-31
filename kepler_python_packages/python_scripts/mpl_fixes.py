"""
Python module conating fixes / overes to mpl library

(under construction)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

from matplotlib.patches import PathPatch, Rectangle, Patch
from matplotlib.text import Text, TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D
#from matplotlib.cbook import is_numlike
from numbers import Number


from matplotlib.transforms import Bbox
from matplotlib.artist import Artist
from matplotlib import rcParams

def is_string_like(obj):
    """
    Return True if *obj* is of string type

    used to avoid iterating over members
    """
    # from depricated matplotlib
    return isinstance(obj, (six.string_types, np.str_, np.unicode_))


class MyText(Text):
    # def __init__(self, *args, **kwargs):
    #    Text.__init__(self, *args, **kwargs)

    # add numeric values for alignemant and offsets

    def __init__(self,
                 x=0, y=0, text='',
                 color=None,          # defaults to rc params
                 verticalalignment='baseline',
                 horizontalalignment='left',
                 verticaloffset = 0.,
                 horizontaloffset = 0.,
                 multialignment=None,
                 fontproperties=None, # defaults to FontProperties()
                 rotation=None,
                 linespacing=None,
                 rotation_mode=None,
                 path_effects=None,
                 usetex=None,          # defaults to rcParams['text.usetex']
                 wrap=False,
                 **kwargs
                 ):
        """
        Create a :class:`~matplotlib.text.Text` instance at *x*, *y*
        with string *text*.

        Valid kwargs are
        %(Text)s
        """

        Artist.__init__(self)
        self._x, self._y = x, y

        if color is None: color = rcParams['text.color']
        if fontproperties is None: fontproperties=FontProperties()
        elif is_string_like(fontproperties): fontproperties=FontProperties(fontproperties)

        self.set_path_effects(path_effects)
        self.set_text(text)
        self.set_color(color)
        self.set_usetex(usetex)
        self.set_wrap(wrap)
        self._verticalalignment = verticalalignment
        self._horizontalalignment = horizontalalignment
        self._verticaloffset = verticaloffset
        self._horizontaloffset = horizontaloffset
        self._multialignment = multialignment
        self._rotation = rotation
        self._fontproperties = fontproperties
        self._bbox = None
        self._bbox_patch = None # a FancyBboxPatch instance
        self._renderer = None
        if linespacing is None:
            linespacing = 1.2   # Maybe use rcParam later.
        self._linespacing = linespacing
        self.set_rotation_mode(rotation_mode)
        self.update(kwargs)
        #self.set_bbox(dict(pad=0))

    def update_from(self, other):
        'Copy properties from other to self'
        Artist.update_from(self, other)
        self._color = other._color
        self._multialignment = other._multialignment
        self._verticalalignment = other._verticalalignment
        self._horizontalalignment = other._horizontalalignment
        self._verticaloffset = other._verticaloffset
        self._horizontaloffset = other._horizontaloffset
        self._fontproperties = other._fontproperties.copy()
        self._rotation = other._rotation
        self._picker = other._picker
        self._linespacing = other._linespacing

    def _get_layout(self, renderer):
        """
        return the extent (bbox) of the text together with
        multile-alignment information. Note that it returns a extent
        of a rotated text when necessary.
        """
        key = self.get_prop_tup()
        if key in self._cached:
            return self._cached[key]

        horizLayout = []

        thisx, thisy  = 0.0, 0.0
        xmin, ymin    = 0.0, 0.0
        width, height = 0.0, 0.0
        lines = self.get_text().split('\n')

        whs = np.zeros((len(lines), 2))
        horizLayout = np.zeros((len(lines), 4))

        if self.get_path_effects():
            from matplotlib.backends.backend_mixed import MixedModeRenderer
            if isinstance(renderer, MixedModeRenderer):
                def get_text_width_height_descent(*kl, **kwargs):
                    return RendererBase.get_text_width_height_descent(renderer._renderer,
                                                                      *kl, **kwargs)
            else:
                def get_text_width_height_descent(*kl, **kwargs):
                    return RendererBase.get_text_width_height_descent(renderer,
                                                                      *kl, **kwargs)
        else:
            get_text_width_height_descent = renderer.get_text_width_height_descent

        # Find full vertical extent of font,
        # including ascenders and descenders:
        tmp, lp_h, lp_bl = get_text_width_height_descent('lp',
                                                         self._fontproperties,
                                                         ismath=False)
        offsety = lp_h * self._linespacing

        baseline = 0
        for i, line in enumerate(lines):
            clean_line, ismath = self.is_math_text(line)
            if clean_line:
                w, h, d = get_text_width_height_descent(clean_line,
                                                        self._fontproperties,
                                                        ismath=ismath)
            else:
                w, h, d = 0, 0, 0

            whs[i] = w, h

            # For general multiline text, we will have a fixed spacing
            # between the "baseline" of the upper line and "top" of
            # the lower line (instead of the "bottom" of the upper
            # line and "top" of the lower line)

            # For multiline text, increase the line spacing when the
            # text net-height(excluding baseline) is larger than that
            # of a "l" (e.g., use of superscripts), which seems
            # what TeX does.

            d_yoffset = max(0, (h-d)-(lp_h-lp_bl))

            horizLayout[i] = thisx, thisy-(d + d_yoffset), \
                             w, h
            baseline = (h - d) - thisy
            thisy -= offsety + d_yoffset
            width = max(width, w)
            descent = d

        ymin = horizLayout[-1][1]
        ymax = horizLayout[0][1] + horizLayout[0][3]
        height = ymax-ymin
        xmax = xmin + width

        # get the rotation matrix
        M = Affine2D().rotate_deg(self.get_rotation())

        offsetLayout = np.zeros((len(lines), 2))
        offsetLayout[:] = horizLayout[:, 0:2]
        # now offset the individual text lines within the box
        if len(lines)>1: # do the multiline aligment
            malign = self._get_multialignment()
            if malign == 'center':
                offsetLayout[:, 0] += width/2.0 - horizLayout[:, 2] / 2.0
            elif malign == 'right':
                offsetLayout[:, 0] += width - horizLayout[:, 2]

        # the corners of the unrotated bounding box
        cornersHoriz = np.array(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)],
            np.float_)
        # now rotate the bbox
        cornersRotated = M.transform(cornersHoriz)

        txs = cornersRotated[:, 0]
        tys = cornersRotated[:, 1]

        # compute the bounds of the rotated box
        xmin, xmax = txs.min(), txs.max()
        ymin, ymax = tys.min(), tys.max()
        width  = xmax - xmin
        height = ymax - ymin

        # Now move the box to the target position offset the display
        # bbox by alignment
        halign = self._horizontalalignment
        valign = self._verticalalignment

        rotation_mode = self.get_rotation_mode()
        if  rotation_mode != "anchor":
            # compute the text location in display coords and the offsets
            # necessary to align the bbox with that location
            if halign=='center':  offsetx = (xmin + width*0.5)
            elif halign=='right': offsetx = (xmin + width)
            #elif is_numlike(halign): offsetx = xmin + halign * (xmax - xmin)
            elif  isinstance(halign, Number): offsetx = xmin + halign * (xmax - xmin)
            else: offsetx = xmin

            if valign=='center': offsety = (ymin + height*0.5)
            elif valign=='top': offsety  = (ymin + height)
            elif valign=='baseline': offsety = (ymin + height) - baseline
            #elif is_numlike(valign): offsety = ymin + valign * (ymax - ymin)
            elif isinstance(valign): offsety = ymin + valign * (ymax - ymin)
            else: offsety = ymin

            offsetx += renderer.points_to_pixels(self._horizontaloffset)
            offsety += renderer.points_to_pixels(self._verticaloffset)
        else:
            xmin1, ymin1 = cornersHoriz[0]
            xmax1, ymax1 = cornersHoriz[2]

            if halign=='center':  offsetx = (xmin1 + xmax1)*0.5
            elif halign=='right': offsetx = xmax1
            #elif is_numlike(halign): offsetx = xmin1 + halign * (xmax1 - xmin1)
            elif isinstance(halign, Number): offsetx = xmin1 + halign * (xmax1 - xmin1)
            else: offsetx = xmin1

            if valign=='center': offsety = (ymin1 + ymax1)*0.5
            elif valign=='top': offsety  = ymax1
            elif valign=='baseline': offsety = ymax1 - baseline
            #elif is_numlike(valign): offsety = ymin1 + valign * (ymax1 - ymin1)
            elif isinstance(valign, Number): offsety = ymin1 + valign * (ymax1 - ymin1)
            else: offsety = ymin1

            offsetx += renderer.points_to_pixels(self._horizontaloffset)
            offsety += renderer.points_to_pixels(self._verticaloffset)

            offsetx, offsety = M.transform_point((offsetx, offsety))


        xmin -= offsetx
        ymin -= offsety

        bbox = Bbox.from_bounds(xmin, ymin, width, height)

        # now rotate the positions around the first x,y position
        xys = M.transform(offsetLayout)
        xys -= (offsetx, offsety)

        xs, ys = xys[:, 0], xys[:, 1]

        ret = bbox, list(zip(lines, whs, xs, ys)), descent
        self._cached[key] = ret
        return ret

    def set_horizontalalignment(self, align):
        """
        Set the horizontal alignment to one of

        ACCEPTS: [ 'center' | 'right' | 'left' | fraction text width from left ]
        """
        legal = ('center', 'right', 'left')
        #if align not in legal and not is_numlike(align):
        if align not in legal and not isinstance(align, Number):
            raise ValueError('Horizontal alignment must be numeric or one of %s' % str(legal))
        self._horizontalalignment = align

    def set_verticalalignment(self, align):
        """
        Set the vertical alignment

        ACCEPTS: [ 'center' | 'top' | 'bottom' | 'baseline' | fraction text height from bottom ]
        """
        legal = ('top', 'bottom', 'center', 'baseline')
        #if align not in legal and not is_numlike(align):
        if align not in legal and not isinstance(align, Number):
            raise ValueError('Vertical alignment must be numeric or one of %s' % str(legal))

        self._verticalalignment = align
