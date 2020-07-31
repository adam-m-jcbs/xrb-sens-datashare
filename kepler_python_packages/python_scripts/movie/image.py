"""
Simple image writer interface.
"""

import os.path

import numpy as np
import PIL

def write_image(arr, filename = 'text.png', format = None):
    # write to file
    img =  PIL.Image.fromarray(arr)
    if format is None:
        format = filename.rsplit('.', 1)[-1]
    filename  = os.path.expanduser(filename)
    filename  = os.path.expandvars(filename)
    if not filename.endswith(format):
        filename = '.'.join((filename, format))
    img.save(
        filename,
        format = format,
        lossless = True,
        quality = 100,
        )

def make_transparent_background(arr, loc = (0, 0), bg = None):
    """
    make transparent background color
    """
    A = np.empty(arr.shape[:-1] + (4,), dtype=arr.dtype)
    A[..., :3] = arr[..., :3]
    A[..., 3] = 255
    if bg is None:
        bg = arr[loc]
    ii = ((arr[..., 0] == bg[0]) &
          (arr[..., 1] == bg[1]) &
          (arr[..., 2] == bg[2]))
    A[ii, 3] = 0
    return A
