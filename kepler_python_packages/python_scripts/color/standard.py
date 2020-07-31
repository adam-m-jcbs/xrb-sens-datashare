from .ifilters import ColorScale
from .functions import ColorBlendBspline, colormap, register_color

cmap_viridis_wrb = ColorBlendBspline(
    ('white',) +
    tuple([ColorScale(colormap('viridis_r'), lambda x: (x-0.2)/0.7)]*2) +
    ('black',), frac=(0,.2,.9,1), k=1)

register_color('viridis_wrb', cmap_viridis_wrb)
