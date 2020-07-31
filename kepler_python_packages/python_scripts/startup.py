# TO LOAD MANUALLY
# import os
# filename = os.environ.get('PYTHONSTARTUP')
# if filename and os.path.isfile(filename):
#     execfile(filename)

# TO LOAD IN IPYTHON in ipy_user_conf.py filename =
#     os.environ.get('PYTHONSTARTUP') if filename and
#     os.path.isfile(filename): execf(filename)

# import physconst
# import kepdump
# import isotope
# import utils
import numpy as np
np.seterr(under='ignore')


# from IPython import get_ipython
# ip = get_ipython()
# try:
#     if ip.has_trait('kernel'):
#         %pylab nbagg
#     %pylab
# except:
#     pass
