
import apng
import io
import os
import os.path
import matplotlib.pylab as plt

def movie_test(n = 11, filename = os.path.expanduser('~/test.png')):
    f = plt.figure(figsize = (6.4,4.8), dpi = 100)
    ax = f.add_subplot(111)
    ani = apng.APNG()
    for i in range(2 * n - 2):
        if i < n:
            ax.plot([i/(n-1),1-i/(n-1)])
        else:
            ax.plot([(i-n+1)/(n-1),(2*n-2-i)/(n-1)],[1,0])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        x = io.BytesIO()
        f.savefig(x, format='png')
        ax.clear()
        # ratio needs to lessim 90 for chrome
        ani.append_file(x, delay=1, delay_den=int(30+10*np.sin(i/n*2*np.pi)))
    ani.save(filename)

import matplotlib.animation as anim

def moviewriter_test(n = 21, filename = os.path.expanduser('~/test.mkv'), dpi = 100):
    fig = plt.figure(figsize = (6.4,4.8), dpi = dpi)
    ax = fig.add_subplot(111)
    moviewriter = anim.FFMpegWriter(
        fps = 30,
        codec = 'libx264rgb', # mkv
        extra_args = '-preset veryslow -crf 0'.split(),
        # codec = 'apng', # apng
        # extra_args = '-plays 0 -preset veryslow'.split(),
        # codec = 'ffv1', # avi
        # extra_args = '-preset veryslow'.split(),
        )
    with moviewriter.saving(fig, filename, dpi):
        for i in range(2 * n - 2):
            ax.clear()
            if i < n:
                ax.plot([i/(n-1),1-i/(n-1)])
            else:
                ax.plot([(i-n+1)/(n-1),(2*n-2-i)/(n-1)],[1,0])
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
            moviewriter.grab_frame()
