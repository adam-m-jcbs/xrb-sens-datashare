"""
some helper tools and common definitions
"""

_Units = ('','k','M','G','T','P','E','Z','Y')
_units = ('','m','u','n','p','f','a')

def _div_lim(x, digits = 0):
    return x * (1 - 2.e-15) - 0.5 * 10**(-digits)
