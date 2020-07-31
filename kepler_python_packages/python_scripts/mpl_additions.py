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
from matplotlib.cbook import is_numlike

from matplotlib.transforms import Bbox, BboxBase

DEBUG = False


class BorderedBbox(BboxBase):
    """
    A :class:`Bbox` that is automatically transformed by a given
    transform.  When either the child bounding box or transform
    changes, the bounds of this bbox will update accordingly.
    """
    def __init__(self, bbox, border, **kwargs):
        """
        *bbox*: a child :class:`Bbox`
        *border*: a child :class:`Bbox`
        """
        assert bbox.is_bbox
        assert border.is_bbox

        BboxBase.__init__(self, **kwargs)
        self._bbox = bbox
        self._border = border
        self.set_children(bbox, border)
        self._points = None

    def __repr__(self):
        return "BorderedBbox(%r, %r)" % (self._bbox, self._border)

    def get_points(self):
        if self._invalid:
            points = self.self._bbox.get_points() - self.self._border.get_points()
            points = np.ma.filled(points, 0.0)
            self._points = points
            self._invalid = 0
        return self._points
    get_points.__doc__ = Bbox.get_points.__doc__

    if DEBUG:
        _get_points = get_points
        def get_points(self):
            points = self._get_points()
            self._check(points)
            return points
