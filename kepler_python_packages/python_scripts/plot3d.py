"""
Python module to plot chart of nuclei.

(under construction)
"""

import numpy as np
from mayavi import mlab
from kepdump import loaddump

def plot3d(model, Xmin=1e-5):
    """
    Plot mass fractions of given model above Xmin fraction
    """
    m = loaddump(model)
    x = m.xb_all[:,1:-1]
    Z = m.Zb
    N = m.Nb

    # Create array with mass fractions
    s = np.zeros((N.max()+1, Z.max()+1, m.jm))
    s[N,Z] = x
    
    src = mlab.pipeline.scalar_field(s)
    volume = mlab.pipeline.volume(        
        src,
        vmin=0, 
        vmax=0.8)

    # mlab.axes()
    mlab.xlabel('N', volume)
    mlab.ylabel('Z', volume)
    mlab.zlabel('Zone', volume)
    mlab.title('Mass fractions for %s'%m.filename, size=2)
    mlab.outline()
    mlab.colorbar(volume, 'Mass fraction', label_fmt='%.1e')
    
    lut = volume.module_manager.scalar_lut_manager.lut
    lut.scale = 'log10'
    lut.range = [Xmin, 1.0]
    
    plane = mlab.pipeline.image_plane_widget(
        src,
        plane_orientation='z_axes',
        slice_index=10)

    lut = plane.module_manager.scalar_lut_manager.lut
    lut.scale = 'log10'
    lut.range = [1.e-10, 1.0]



#    mlab.savefig('/tmp/test.eps')
    return volume

