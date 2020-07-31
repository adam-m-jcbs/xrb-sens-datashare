import numpy as np

from .base import Story

from .animations import RotationAnimation

from rotation import ex, ey, ez, rotate

class RotationStory(Story):
    """
    Three different ways to rotate around an axis.
    """
    _ncyc = 3

    pos1 = rotate(ex, ez * np.arcsin(-1/_ncyc))
    _animations = [
        [RotationAnimation, dict(
             pos1 = pos1,
             w = ey,
             rot_mode = 'w',
             )
         ],
        [RotationAnimation, dict(
             pos1 = pos1,
             w = ey,
             rot_mode = 'inertial',
             )
         ],
        [RotationAnimation, dict(
             pos1 = pos1,
             w = ey,
             rot_mode = 'project',
             ncyc = _ncyc,
             )
         ],
        ]


class RotVecStory(Story): pass
