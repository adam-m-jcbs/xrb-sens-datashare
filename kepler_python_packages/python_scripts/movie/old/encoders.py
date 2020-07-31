import apng
import io
import os
import os.path
import multiprocessing
import threading
import os.path
import numpy as np
import PIL
import subprocess

from matplotlib.animation import FFMpegWriter

def queue_tee(qi, *qo):
    while True:
        item = qi.get()
        for q in qo:
            q.put(item)
        qi.task_done()
        if item is None:
            break

def write_movie(frames, filename, format = None, **kwargs):
    if isinstance(filename, tuple):
        n = len(filename)
        if not isinstance(format, tuple):
            format = (format,) * n
        if isinstance(frames, multiprocessing.queues.Queue):
            qo = [multiprocessing.JoinableQueue() for _ in range(n)]
            tee = threading.Thread(
                target = queue_tee,
                args = (frames, *qo),
                )
            tee.start()
            processes = []
            for i in range(n):
                processes.append(multiprocessing.Process(
                    target = write_movie,
                    args = (qo[i], filename[i], format[i]),
                    kwargs = kwargs,
                    ))
                processes[-1].start()
            for p in processes:
                p.join()
            tee.join()
        else:
            for fn,fm in zip(filename, format):
                write_movie(frames, fn, format = fm, **kwargs)
        return
    if format is None:
        format = filename.rsplit('.', 1)[-1]
    if format in ('gif', 'tif', 'tiff', 'webp' ):
        write_PILLOW_movie(frames, filename, format = format, **kwargs)
    elif format in ('png', 'apng'):
        write_APNG_movie(frames, filename, format = format, **kwargs)
    elif format in ('mkv', 'avi', 'webm', 'vp8', 'flv', 'mp4', 'mov', ):
        write_FFMPEG_movie(frames, filename, format = format, **kwargs)
    elif filename is None:
        write_no_movie(frames, filename, format = format, **kwargs)
    else:
        raise AttributeError(f'Unknown file format "{format}".')

def write_no_movie(frames, filename, format = None, **kwargs):
    print(' [WARNING] Writing no movie. ')
    if isinstance(frames, multiprocessing.queue):
        while True:
            q = frames.get()
            try:
                q.task_done()
            except:
                pass
            if q is None:
                break

class MovieWriterProcess(multiprocessing.Process):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
    def run(self):
        write_movie(*self.args, **self.kwargs)

ADD_WRITER = 'add_writer'
DEL_WRITER = None

def queue_frame_sorter(qi, qo):
    print('[Sorter] Starting.')
    store = dict()
    nframe = -1
    nwriter = 0
    while True:
        item = qi.get()
        if isinstance(item, str) and item == ADD_WRITER:
            nwriter += 1
            print(f'[Sorter] Adding Writer {nwriter}.')
        elif item is None:
            assert nwriter > 0
            print(f'[Sorter] Removing Writer {nwriter}.')
            nwriter -= 1
        else:
            if isinstance(item, (tuple, list)):
                assert len(item) == 2
                if np.shape(item[0]) == ():
                    iframe, frame = item
                else:
                    frame, iframe = item
                assert np.shape(iframe) == ()
            else:
                iframe = nframe + 1
                frame = item
            assert len(np.shape(frame)) == 3
            assert iframe > nframe
            assert iframe not in store
            store[iframe] = frame
            print(f'[Sorter] Achiving Frame {iframe} ({len(store)}).')
        while True:
            jframe = nframe + 1
            if jframe in store:
                qo.put(store.pop(jframe))
                print(f'[Sorter] Sending Frame {jframe} ({len(store)}).')
                nframe = jframe
            else:
                break
        qi.task_done()
        if nwriter == 0:
            assert len(store) == 0
            break
    qo.put(None)
    print(f'[Sorter] Done {nframe+1} frames.')

class MultiMovieWriterProcess(MovieWriterProcess):
    def __init__(self, frames, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames = frames
    def run(self):
        qo = multiprocessing.JoinableQueue()
        sorter = threading.Thread(
            target = queue_frame_sorter,
            args = (self.frames, qo),
            )
        sorter.start()
        self.args = (qo,) + self.args
        super().run()


#=======================================================================

def process_frame_to_PIL(qi, qo):
    while True:
        f = qi.get()
        if f is not None:
            f = Image.fromarray(f)
        qo.put(f)
        qi.task_done()
        if f is None:
            break

def write_PILLOW_movie(
        frames,
        filename = 'xxx',
        format = 'webp',
        delay = 1/60,
        loop = 0,
        collect = False,
        **kwargs,
        ):

    # write to file
    filename  = os.path.expanduser(filename)
    filename  = os.path.expandvars(filename)
    if not filename.endswith(format):
        filename = '.'.join((filename, format))
    delay = int(delay * 1000)

    if isinstance(frames, multiprocessing.queues.Queue):
        frame0 = frames.get()
        frameq = multiprocessing.JoinableQueue()
        thread = threading.Thread(
            target = process_frame_to_PIL, args = (frames, frameq))
        thread.start()
        frames = iter(frameq.get, None)

        # buffer if requested
        if collect == True:
            frames = [f for f in frames]
    else:
        frame0 = frames[0]
        frames = [Image.fromarray(a) for a in frames[1:]]
    frame0 = Image.fromarray(frame0)

    extra_args = dict()
    if format in ('gif', ):
        extra_args['optimize'] = True
        extra_args['interlace'] = False
        extra_args['comment'] = 'Animated GIF'
        if delay < 20:
            print(f' [GIF] Does not support delay < 20 ms (provided: {delay})')
        if not isinstance(extra_args['comment'], bytes):
            extra_args['comment'] = extra_args['comment'].encode('US-ASCII', errors = 'replace')
    elif format in ('tif', 'tiff',):
        # PIL.TiffImagePlugin.COMPRESSION_INFO
        # https://en.wikipedia.org/wiki/TIFF
        # 'raw' - no compression
        # 'jpeg' - with loss
        extra_args['compression'] = 'tiff_lzw'
    elif format in ('webp', ):
        extra_args['quality'] = 100
        extra_args['minimize_size'] = True
        extra_args['lossless'] = True
        extra_args['method'] = 6
    else:
        raise AttributeError(f'Format {format} not supprted.')
    frame0.save(
        filename,
        format = format,
        save_all = True,
        duration = delay,
        loop = loop,
        append_images = frames,
        **extra_args,
        )

def write_APNG_movie(
        frames,
        filename = 'xxx.png',
        format = 'png',
        delay = 1/60,
        loop = 0,
        **kwargs,
        ):
    # write to file
    filename  = os.path.expanduser(filename)
    filename  = os.path.expandvars(filename)
    ani = apng.APNG()
    if isinstance(frames, multiprocessing.queues.Queue):
        frameq = multiprocessing.JoinableQueue()
        thread = threading.Thread(
            target = process_frame_to_PIL, args = (frames, frameq))
        thread.start()
        frames = iter(frameq.get, None)
    else:
        frames = [Image.fromarray(frame) for frame in frames]
    for frame in frames:
        x = io.BytesIO()
        frame.save(
            x,
            format = 'png',
            lossless = True,
            quality = 100,
        )
        ani.append_file(x, delay=1, delay_den=int(1/delay))
    ani.save(filename)

class PhonyFig(object):
    def __init__(self, frames, dpi = 100):
        self.dpi = dpi
        self.frames = frames
        if isinstance (frames, multiprocessing.queues.Queue):
            self.frame = self.frames.get()
            self.frames.task_done()
        else:
            self.frame = self.frames[0]
        if self.frame is None:
            self.size = np.array([0,0])
        else:
            self.size = np.asarray(self.frame.shape[1::-1])
        self.size_inches = self.size / self.dpi
        self.pos = 0

    def get_size_inches(self):
        return self.size_inches

    def set_size_inches(self, *args, **kwargs):
        # maybe need to adjust dpi
        pass

    # this is an internal routine use by this overwrite
    def __get_frame(self):
        """
        return next frame
        """
        self.pos += 1
        frame = self.frame
        if isinstance(self.frames, multiprocessing.queues.Queue):
            self.frame = self.frames.get()
            self.frames.task_done()
        else:
            if self.pos < len(self.frames):
                self.frame = self.frames[self.pos]
            else:
                self.frame = None
        return frame

    def savefig(self, f, *args, **kwargs):
        dpi = kwargs.get('dpi', self.dpi)
        assert dpi == self.dpi
        format = kwargs.get('format')
        frame = self.__get_frame()
        assert isinstance(f, io.BufferedWriter)
        assert format == 'rgba'
        assert frame.shape[2:] == (4,)
        f.write(frame)

class BufferFFMpegWriter(FFMpegWriter):
    """
    mixin class to replace figures by frames from array or queue

    see
    ~/Python/lib/python3.7/site-packages/matplotlib/animation.py
    """
    def setup(self, stream, filename, *args, **kwargs):
        super().setup(PhonyFig(stream, *args, **kwargs), filename, *args, **kwargs)
        if self.fig.frame is None:
            return

    def save_frames(self, frames, filename, dpi = 100):
        with self.saving(frames, filename, dpi):
            while self.fig.frame is not None:
                self.grab_frame()

def ffmpeg_defaults(format, **kwargs):
    extra_args = list()
    if format in ('mp4', 'mpeg', ):
        # '-crf 0' is lossless
        # '-pix_fmt yuv420p10le' for 10 bit encoding
        # '-crf xxx' with xxx = 18-22 is good quality
        # codec = 'libx264rgb'  # rgb input
        xcodec = kwargs.get('codec', 'h264')
        extra_args = []
        if xcodec == 'h265':
            # h265 - seems not supported by default
            # https://trac.ffmpeg.org/wiki/CompilationGuide/Centos
            codec = 'libx265' # converts to video color space
            crf = kwargs.get('crf', 23) # h265 allows 5 lower crf at same quality
            extra_args.extend(f'-preset veryslow -tune animation -crf {crf}'.split())
            #extra_args.extend(f'-preset veryslow -crf {crf}'.split())
            #extra_args.extend('-loglevel verbose'.split())
            extra_args.extend('-tag:v hvc1'.split())
        else:
            # h264
            codec = 'libx264' # converts to video color space
            crf = kwargs.get('crf', 18)
            extra_args.extend(f'-preset veryslow -tune animation -crf {crf}'.split())
        extra_args.extend('-pix_fmt yuv420p'.split())
        #extra_args.extend('-pix_fmt yuv420p10le'.split())
        #print(f'using {codec} {extra_args}')
    elif format in ('mkv', 'flv', ):
        # '-crf 0' is lossless
        codec = 'libx264rgb'
        extra_args = '-preset veryslow -tune animation -crf 0'.split()
    elif format == 'avi':
        codec = 'ffv1'
        extra_args = '-preset veryslow'.split()
    elif format in ('webm', ):
        codec = 'libvpx-vp9'
        if kwargs.get('lossless', False):
            extra_args = '-lossless 1 -deadline best'.split()
        else:
            crf = kwargs.get('crf', 30)
            extra_args = f'-crf {crf} -b:v 0 -deadline best'.split()
        # extra_args.extend(list('-pix_fmt yuv420p'.split()))
        extra_args.extend(list('-threads 16 -row-mt 1'.split()))
    elif format in ('vp8', ):
        # seems not to be working
        codec = 'libvpx'
        # extra_args = '-pix_fmt yuva420p -metadata:s:v:0 alpha_mode="1"'.split()
        extra_args = '-pix_fmt yuva420p'.split()
    elif format in ('webp', ):
        # does not seem to work
        codec = 'libwebp'
        extra_args = '-lossless 1 -compression_level 6 -preset drawing -pix_fmt rgba'.split()
    elif format in ('mov', ):
        codec = 'libx265' # converts to video color space
        crf = kwargs.get('crf', 23) # h265 allows 5 lower crf at same quality
        extra_args.extend(f'-preset veryslow -tune animation -crf {crf}'.split())
        extra_args.extend('-tag:v hvc1'.split())
        extra_args.extend('-pix_fmt yuv420p'.split())
        extra_args.extend('-bsf:v hevc_mp4toannexb'.split())
    else:
        raise AttributeError('Unsupported file format')
    return codec, extra_args


def write_FFMPEG_movie_mpl(
        frames,
        filename = 'xxx.mp4',
        format = 'mp4',
        delay = 1/60,
        dpi = 100,
        **kwargs,
        ):
    filename  = os.path.expanduser(filename)
    filename  = os.path.expandvars(filename)
    if format is None:
        format = filename.rsplit('.', 1)[-1]

    codec, extra_args = ffmpeg_defaults(format, **kwargs)

    moviewriter = BufferFFMpegWriter(
        fps = 1/delay,
        codec = codec,
        extra_args = extra_args,
        )
    moviewriter.save_frames(frames, filename)


# driver adopted from matplotlib
def write_FFMPEG_movie(
        frames,
        filename = 'xxx.mp4',
        format = 'mp4',
        delay = 1/60,
        dpi = 100,
        bitrate = 0,
        exec_path = '/usr/bin/ffmpeg',
        metadata = dict(),
        **kwargs,
        ):
    filename  = os.path.expanduser(filename)
    filename  = os.path.expandvars(filename)
    if format is None:
        format = filename.rsplit('.', 1)[-1]

    codec, extra_args = ffmpeg_defaults(format, **kwargs)
    fps = int(1/delay)

    if isinstance(frames, multiprocessing.queues.Queue):
        frame = frames.get()
    else:
        frame = frames[0]
    frame_size = frame.shape[1::-1]

    args = [exec_path,
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{frame_size[0]}x{frame_size[1]}',
            '-pix_fmt', 'rgba',
            '-r', f'{fps}',
            '-loglevel', 'quiet',
            '-i', 'pipe:']

    args.extend(['-c:v', codec])

    if bitrate > 0:
        args.extend(['-b', f'{bitrate:d}k'])
    if extra_args is not None:
        args.extend(extra_args)
    for k, v in metadata.items():
        args.extend(['-metadata', f'{v}={k}'])
    args.extend(['-y', f'{filename}'])

    print('[write_FFMPEG_movie] ' + ' '.join(args))

    output = subprocess.PIPE
    proc = subprocess.Popen(
        args,
        shell = False,
        stdout = output,
        stderr = output,
        stdin = subprocess.PIPE,
        creationflags = 0)
    nframe = 0
    starttime = time.time()
    if isinstance(frames, multiprocessing.queues.Queue):
        while frame is not None:
            nframe += 1
            runtime = time.time() - starttime
            eta = runtime / nframe * frames.qsize()
            print(f'[write_FFMPEG_movie] writing frame {nframe} (runtime: {time2human(runtime)}, ETA: {time2human(eta)}).')
            proc.stdin.write(frame)
            frame = frames.get()
    else:
        for frame in frames:
            nframe += 1
            print(f'[write_FFMPEG_movie] writing frame {nframe} ({len(frames)}).')
            proc.stdin.write(frame)

    out, err = proc.communicate()
    proc.stdin.close()

    if len(out) > 0:
        print(f'\n[FFMPEG] {out.decode()}')
    if len(err) > 0:
        print(f'\n[FFMPEG] {err.decode()}')
