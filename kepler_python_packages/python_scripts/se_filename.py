"""
Modules provides routine to generate se file names.
"""

def se_filename(mass = 1.e0,\
                metallicity = 0.02e0,\
                comment = "standard",\
                cycle = 0):
    """Generate SE file name."""
    return "M{M:06g}Z{Z:07g}.{c:s}.{C:07d}.se.h5".format(\
            M = float(mass),\
            Z = float(metallicity),\
            c = comment.strip(),\
            C = int(cycle))
