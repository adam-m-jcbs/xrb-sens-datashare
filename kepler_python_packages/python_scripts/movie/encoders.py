"""
Movie Encoder Library
"""
from multiprocessing import Process
from multiprocessing import JoinableQueue
from multiprocessing import Queue
from multiprocessing.queues import Queue as Queue_type
from threading import Thread
from io import BytesIO
from io import BufferedWriter
from time import time
from os.path import expanduser
from os.path import expandvars
from subprocess import PIPE
from subprocess import Popen

from numpy import ndim
from numpy import asarray
from matplotlib.animation import FFMpegWriter
from PIL.Image import fromarray
from apng import APNG

from human import time2human

from .frames import Frame
from .raw import MemoryMovie
from .raw import RawMovieWriter
from .raw import RawMovieReader

ADD_WRITER = 'add_writer'
DEL_WRITER = None

ENC_CLOSE = None
ENC_CALL = 'WRITE'
ENC_DONE = 'DONE'
ENC_ERROR = 'ERROR'
ENC_FRAME = 'FRAME'
ENC_UNKOWN = 'UNKNOWN'

def encode(rawfile, outfiles, **kwargs):
    with Encoder(outfiles, **kwargs) as encoder, \
             RawMovieReader(rawfile) as source:
        for i, frame in enumerate(source):
            encoder.write_frame(Frame(arr=frame, iframe=i))

class Encoder(object):
    """
    This is the master encoder that calls the individual encoders based on file names(s)
    """
    def __init__(
            self,
            filename,
            format= None,
            parallel = True,
            sort = False,
            **kwargs,
            ):

        framerate = kwargs.pop('framerate', None)
        delay = kwargs.get('delay', None)
        if delay is None and framerate is not None:
            delay = 1 / framerate
            kwargs['delay'] = delay

        if isinstance(filename, tuple):
            n = len(filename)
            if not isinstance(format, tuple):
                formats = (format,) * n
            filenames = filename
        else:
            filenames = (filename,)
            formats = (format,)

        self.sort = sort
        if self.sort:
            self.nframe = -1
            self.store = dict()

        self.writers = []

        _is_process = kwargs.pop('_is_process', False)
        if parallel and not (_is_process and len(filenames) == 1):
            for filename,format in zip(filenames, formats):
                self.writers.append(
                    EncoderClient(
                        filename,
                        format=format,
                        parallel = False,
                        sort = False,
                        **kwargs,
                        ))
            return

        for filename, format in zip(filenames, formats):
            if format is None:
                format = filename.rsplit('.', 1)[-1]
            if format in ('gif', 'tif', 'tiff', 'webp', ):
                self.writers.append(
                    PillowEncoder(filename, format = format, **kwargs))
            elif format in ('png', 'apng'):
                self.writers.append(
                    APNGEncoder(filename, format = format, **kwargs))
            elif format in ('mkv', 'avi', 'webm', 'vp8', 'flv', 'mp4', 'mov', 'mpg', ):
                if kwargs.pop('use_mpl', False):
                    self.writers.append(
                        FFMPEGEncoderMPL(filename, format = format, **kwargs))
                else:
                    self.writers.append(
                        FFMPEGEncoder(filename, format = format, **kwargs))
            elif (format in ('raw', 'rgz', 'rxz', 'rbz') or
                    filename.endswith(
                        ('raw', 'raw.gz', 'raw.xz', 'raw.bz', 'rgz', 'rxz', 'rbz'))):
                self.writers.append(
                    RawEncoder(filename, format = format, **kwargs))
            elif (format in ('mem', 'mem_gz', 'mem_xz', 'mem_bz') or
                    filename in ('mem', 'mem.gz', 'mem.xz', 'mem.bz')):
                self.writers.append(
                    MemoryEncoder(filename, format = format, **kwargs))
            elif filename is None:
                self.writers.append(
                    NoEncoder(filename, format = format, **kwargs))
            else:
                raise AttributeError(f'Unknown file format "{format}".')

    def write(self, frames, batch = None):
        """
        write all frames then close unless in batch mode
        """
        iframe = 0
        starttime = time()
        if isinstance(frames, Queue_type):
            # since the queue has to come from a separate process,
            # this does not work well in applications implemented here
            frame = frames.get()
            while frame is not None:
                iframe += 1
                runtime = ime() - starttime
                jframe = frames.qsize()
                eta = runtime / iframe * jframe
                print(f'[{self.__class__.__name__}] writing frame {iframe} (runtime: {time2human(runtime)}, queue: {jframe}, ETA: {time2human(eta)}).')
                self._write_frame(frame)
                frame = frames.get()
        else:
            if isinstance(frames, Frame):
                if batch is None:
                    batch = True
                frames = (frames,)
            nframe = len(frames)
            for i, frame in enumerate(frames):
                iframe = i + 1
                jframe = nframe - iframe
                runtime = time() - starttime
                eta = runtime / nframe * jframe
                print(f'[{self.__class__.__name__}] writing frame {iframe} (runtime: {time2human(runtime)}, total: {nframe}, ETA: {time2human(eta)}).')
                self._write_frame(frame)
        if batch is None:
            batch = False
        if not batch:
            self.close()

    def reset_sort_count(self, start = 0):
        assert len(self.store) == 0
        self.nframe = start - 1

    def write_frame(self, frame):
        """
        include sorting
        """
        if not self.sort:
            self._write_frame(frame)
            return
        assert isinstance(frame,  Frame)
        iframe = frame.oframe
        if iframe is None:
            iframe = frame.iframe
        if iframe is None:
            if len(self.store) == 0:
                iframe = self.nframe + 1
            else:
                iframe = max(self.store.keys()) + 1
        assert iframe > self.nframe
        assert iframe not in self.store
        self.store[iframe] = frame
        print(f'[{self.__class__.__name__}] Achiving Frame {iframe} ({len(self.store)}).')
        while True:
            jframe = self.nframe + 1
            if jframe in self.store:
                super()._write_frame(self.store.pop(jframe))
                print(f'[{self.__class__.__name__}] Sending Frame {jframe} ({len(self.store)}).')
                nframe = jframe
            else:
                break

    def _write_frame(self, frame):
        """
        write single frame, do not close file
        """
        for w in self.writers:
            if isinstance(w, EncoderClient):
                w.write_frame(frame)
            else:
                w.write_frame(frame.arr)

    def close(self):
        if self.sort:
            assert len(self.store) == 0
        for w in self.writers:
            w.close()
        self.writers = []

    def get_qsize(self):
        qsize = None
        for w in self.writers:
            qs = w.get_qsize()
            if qs is not None:
                if qsize is None:
                    qsize = 0
                qsize = qsize + qs
        return qsize

    def __enter__(self, *args, **kwargs):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

class EncoderProcess(Process):
    """
    Start encoding in separate process
    """
    def __init__(self, qc, *args, **kwargs):
        super().__init__()
        self.qc = qc
        self.args = args
        self.kwargs = kwargs
        self.qr = kwargs.pop('qr', None)
    def run(self):
        self.kwargs['_is_process'] = True
        self.enc = Encoder(*self.args, **self.kwargs)
        while True:
            task = self.qc.get()
            if task[0] == ENC_CLOSE:
                print(f'[{self.__class__.__name__}] Closing ... ')
                self.enc.close()
                self.qc.task_done()
                break
            if task[0] == ENC_FRAME:
                frame = task[1]
                self.enc.write_frame(frame)
            elif task[0] == ENC_CALL:
                args = task[2]
                kwargs = task[3]
                func = getattr(self.enc, task[1], None)
                if func is not None:
                    result = func(*args, **kwargs)
                    if self.qr is not None:
                        self.qr.put(result)
                    elif result is not None:
                        print(f'[{self.__class__.__name__}] Output: {result}')
                else:
                    if self.qr is not None:
                        self.qr.put(ENC_ERROR)
                    else:
                        print(f'[{self.__class__.__name__}] ERROR: no {task[0]}')
            else:
                if self.qr is not None:
                    self.qr.put(ENC_UNKNOWN)
                else:
                    print(f'[{self.__class__.__name__}] Unkonwn task: {task}')
            self.qc.task_done()
        if self.qr is not None:
            self.qr.put(ENC_DONE)
        else:
            print(f'[{self.__class__.__name__}] Done.')

class EncoderClient(object):
    """
    local interface that mirrors communication from encoder
    """
    def __init__(self, *args, **kwargs):
        self.qc = JoinableQueue()
        if kwargs.pop('return', False):
            self.qr = JoinableQueue()
            kwargs['qr'] = self.qr
        else:
            self.qr = None
        self.proc = EncoderProcess(self.qc, *args, **kwargs)
        self.proc.start()
        self.wq = None

    def __getattr__(self, func):
        if func in dir(Encoder):
            if self.qc is None:
                print(f'[{self.__class__.__name__}] Encoder queue already closed.')
            else:
                def remote(*args, **kwargs):
                    self.qc.put((ENC_CALL, func, args, kwargs))
                    if self.qr is not None:
                        return self.qr.get()
                return remote
        raise AttributeError()

    def write(self, frames, batch = False, **kwargs):
        # It might be possible, but would need to be tested, to pass a
        # Queue proxy from a manager through the queue.
        if self.qc is None:
            print(f'[{self.__class__.__name__}] Encoder command queue already closed.')
            return
        kwargs['batch'] = batch
        if self.wq is not None:
            if not batch:
                raise Exception('Queue already running.')
            print(f'[{self.__class__.__name__}] Waiting for previous queue to terminate.')
            self.wq.join()
            self.wq = None
        if isinstance(frames, Queue_type):
            def pipe(qi, qo, batch = False):
                nframe = 0
                print('[pipe] Waiting for first frame ...')
                while True:
                    nframe += 1
                    frame = qi.get()
                    if frame is not None:
                        qo.put((ENC_FRAME, frame))
                    qi.task_done()
                    if frame is None:
                        break
                    print(f'[pipe] got frame {nframe}')
                print('[pipe] Done.')
            self.wq = Process(target = pipe, args = (frames, self.qc, batch))
            self.wq.start()
        else:
            self.qc.put((ENC_CALL, 'write', (frames,) , kwargs))
            if self.qr is not None:
                return self.qr.get()
        if not batch and self.wq is None:
            self.close()

    def get_qsize(self):
        if self.qc is not None:
            return self.qc.qsize()
        return None

    def close(self):
        if self.wq is not None:
            print(f'[{self.__class__.__name__}] Waiting for queue to terminate.')
            self.wq.join()
            self.wq = None
        if self.qc is None:
            return
        self.qc.put((ENC_CLOSE,))
        if self.qr is not None:
            result = self.qr.get()
            if result is not None:
                print('[{self.__class__.__name__}] - {result}')
        print(f'[{self.__class__.__name__}] Waiting for Encoder Command Queue to finish ...')
        self.qc.join()
        self.qc = None
        print(f'[{self.__class__.__name__}] Waiting for EncoderProcess to finish ...')
        self.proc.join()
        print(f'[{self.__class__.__name__}] Done.')

class QueueFrameSorterBase(object):
    """
    TODO - synchronisation of mutiple writes by
           - specifing # of frames (could be at end)
           - sortable sequence ID (could be, e.g., time.time())
           - specify number of items in sequence rather than number of writers
    """
    def __init__(self, qi, qo, num = 'iframe', batch = False):
        super().__init__()
        self.qi = qi
        self.qo = qo
        self.batch = batch
        self.num = num
    def run(self):
        store = dict()
        nframe = -1
        nwriter = 0
        while True:
            item = self.qi.get()
            if isinstance(item, str) and item == ADD_WRITER:
                nwriter += 1
                print(f'[{self.__class__.__name__}] Adding Writer {nwriter}.')
            elif item is None:
                if nwriter > 0:
                    print(f'[{self.__class__.__name__}] Removing Writer {nwriter}.')
                    nwriter -= 1
                elif self.batch:
                    self.qi.task_done()
                    break
                else:
                    raise Exception('[Sorter] Too many None.')
            else:
                frame = item
                assert isinstance(frame,  Frame)
                if self.num == 'iframe':
                    iframe = frame.iframe
                else:
                    iframe = frame.oframe
                if iframe is None:
                    iframe = nframe + 1
                assert iframe > nframe
                assert iframe not in store
                store[iframe] = frame
                print(f'[{self.__class__.__name__}] Achiving Frame {iframe} ({len(store)}).')
            while True:
                jframe = nframe + 1
                if jframe in store:
                    self.qo.put(store.pop(jframe))
                    print(f'[{self.__class__.__name__}] Sending Frame {jframe} ({len(store)}).')
                    nframe = jframe
                else:
                    break
            self.qi.task_done()
            if nwriter == 0:
                assert len(store) == 0
                if not self.batch:
                    break
                self.nframe = -1
        self.qo.put(None)
        print(f'[{self.__class__.__name__}] Done {nframe+1} frames.')

class QueueFrameSorterThread(QueueFrameSorterBase, Thread):
    pass
class QueueFrameSorterProcess(QueueFrameSorterBase, Process):
    pass

class QueueFrameSorter(object):
    def __init__(self, qi = None, qo = None, batch = False, num = 'iframe',
                 process = False):
        self.batch = batch
        if qi is None:
            qi = JoinableQueue()
            self.own_qi = True
        else:
            self.own_qi = False
        if qo is None:
            qo = JoinableQueue()
            self.own_qo = True
        else:
            self.own_qo = False
        if process:
            self.sorter = QueueFrameSorterProcess(qi, qo, num, batch)
        else:
            self.sorter = QueueFrameSorterThread(qi, qo, num, batch)
        self.sorter.start()
        self.qi = qi
        self.qo = qo
        self.num = num
    def get_output_queue(self):
        return self.qo
    def get_input_queue(self):
        return self.qi
    def get_queues(self):
        return self.qi, self.qo
    def close(self):
        if self.sorter.batch:
            self.qo.put(None)
        if self.own_qo:
            self.qo.close()
        if self.own_qi:
            self.qi.close()
        self.sorter.join()

#=======================================================================
# actual encoders

class BaseEncoder(object):
    closed = object()
    def __init__(self):
        self.nframe = 0
        self.starttime = time()
    def write_frame(self, frame):
        """
        write single frame
        """
        self.nframe += 1
        print(f'[{self.__class__.__name__}] Wrote frame {self.nframe}')
    def close(self):
        """
        close file.
        """
        xtime = time2human(time() - self.starttime)
        print(f'[{self.__class__.__name__}] Wrote {self.nframe} frames in {xtime}.')
    def get_qsize(self):
        """
        Nothing certain is not known
        """
        return None
    def get_written(self):
        return self.nframe


class NoEncoder(BaseEncoder):
    """
    Just discard all stuff
    """
    def get_qsize(self):
        return 0


class FileEncoder(BaseEncoder):
    def __init__(self, filename):
        super().__init__()
        filename  = expanduser(filename)
        filename  = expandvars(filename)
        self.filename = filename

class MemoryEncoder(BaseEncoder):
    """
    Just write to Memory

    usually a MemoryMovie object should be passed
    """
    def __init__(self, filename = None, format = None, **kwargs):
        super().__init__()
        self.mem = kwargs.get('mem', None)
        if self.mem is None:
            if isinstance(format, str) and format.count('_') == 1:
                compression = filename.rsplit('_', 1)[-1]
                if not compression in ('xz', 'gz', 'bz'):
                    raise AttributeError(f'Unknown compression {compression}')
            elif isinstance(filename, str) and filename.count('.') == 1:
                compression = filename.rsplit('.', 1)[-1]
                if not compression in ('xz', 'gz', 'bz'):
                    raise AttributeError(f'Unknown compression {compression}')
            else:
                compression = None
            self.mem = MemoryMovie(compression = compression, **kwargs)
        else:
            assert isinstance(self.mem, MemoryMovie)
    def close(self):
        # does nothoing
        self.mem.close()
    def write_frame(self, frame):
        self.mem.write(frame)
    def get_qsize(self):
        return 0

class RawEncoder(FileEncoder):
    """
    Just write to Raw File

    add append option
    """
    def __init__(self, filename = None, format = None, **kwargs):
        self.raw = kwargs.get('raw', None)
        self.append = kwargs.get('append', None)
        if self.raw is None:
            self.raw = RawMovieWriter(filename, format=format, **kwargs)
        else:
            assert isinstance(self.mem, MemoryMovie)
    def close(self):
        self.raw.close()
    def write_frame(self, frame):
        self.raw.write(frame)
    def get_qsize(self):
        return 0

class FFMPEGDefaults(object):
    """Mixin class"""
    @staticmethod
    def defaults(format, **kwargs):
        extra_args = list()
        if format in ('mpeg', 'mpg', ):
            codec = 'mpeg2video'
            extra_args = '-pix_fmt yuva420p -threads 16 -qscale:v 2 -video_format component'.split()
        elif format in ('mp4', ):
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

class FFMPEGEncoder(FileEncoder, FFMPEGDefaults):
    def __init__(
            self,
            filename = 'xxx.mp4',
            format = None,
            delay = 1/60,
            **kwargs,
            ):
        super().__init__(filename)
        self.format = format
        self.fps = int(1/delay)
        self.kwargs = kwargs
        self.proc = None

    def start_writer(self, frame_size):
        bitrate = self.kwargs.get('bitrate', 0)
        exec_path = self.kwargs.get('exec_path', '/usr/bin/ffmpeg') # adopt for OS
        metadata = self.kwargs.get('metadata', dict())

        format = self.format
        if format is None:
            format = self.filename.rsplit('.', 1)[-1]
        codec, extra_args = self.defaults(format, **self.kwargs)

        fps = self.fps
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
        args.extend(['-y', self.filename])

        print(f'[{self.__class__.__name__}] ' + ' '.join(args))

        output = PIPE
        pinput = PIPE
        self.proc = Popen(
            args,
            shell = False,
            stdout = output,
            stderr = output,
            stdin = pinput,
            creationflags = 0)

    def write_frame(self, frame):
        if self.proc is self.closed:
            print(f'[{self.__class__.__name__}] Aleady closed.')
            return
        if self.proc is None:
            frame_size = frame.shape[1::-1]
            self.start_writer(frame_size)
        self.proc.stdin.write(frame)
        super().write_frame(frame)

    def close(self):
        if self.proc is None:
            print(f'[{self.__class__.__name__}] Never opened.')
            return
        if self.proc is self.closed:
            print(f'[{self.__class__.__name__}] Aleady closed.')
            return
        out, err = self.proc.communicate()
        self.proc.stdin.close()
        self.proc = self.closed

        if len(out) > 0:
            print(f'\n[{self.__class__.__name__}] {out.decode()}')
        if len(err) > 0:
            print(f'\n[{self.__class__.__name__}] {err.decode()}')

        super().close()
        print(f'[{self.__class__.__name__}] Done.')

class APNGEncoder(FileEncoder):
    def __init__(
            self,
            filename = 'xxx.png',
            format = 'png',
            delay = 1/60,
            loop = 0,
            lossless = True,
            quality = 100,
            ):
        super().__init__(filename)
        self.proc = APNG(num_plays = loop)
        self.fps = int(1/delay)
        self.quality = quality
        self.lossless = lossless
        assert format == 'png'

    def write_frame(self, frame):
        if self.proc is self.closed:
            print(f'[{self.__class__.__name__}] Aleady closed.')
            return
        frame = fromarray(frame)
        memfile = BytesIO()
        frame.save(
            memfile,
            format = 'png',
            lossless = self.lossless,
            quality = self.quality,
        )
        self.proc.append_file(memfile, delay=1, delay_den=self.fps)
        super().write_frame(frame)

    def close(self):
        if self.proc is self.closed:
            print(f'[{self.__class__.__name__}] Aleady closed.')
            return
        self.proc.save(self.filename)
        self.proc = self.closed
        super().close()
        print(f'[{self.__class__.__name__}] Done.')


class PillowEncoder(FileEncoder):
    def __init__(
            self,
            filename = 'xxx.webp',
            format = None,
            delay = 1/60,
            collect = False,
            **kwargs,
            ):
        super().__init__(filename)
        self.delay = int(delay * 1000)
        self.format = format
        self.kwargs = kwargs
        self.collect = collect
        self.proc = None
        if collect:
            self.frames = list()
        else:
            self.frames = Queue()

    def start_writer(self, frame0):

        format = self.format
        if format is None:
            format = self.filename.rsplit('.', 1)[-1]

        kwargs = dict()
        kwargs['loop'] = self.kwargs.get('loop', 0)
        kwargs['format'] = format
        kwargs['save_all'] = True
        kwargs['delay'] = self.delay
        args = (self.filename,)

        if format in ('gif', ):
            kwargs['optimize'] = self.kwargs.get('optimize', True)
            kwargs['interlace'] = self.kwargs.get('interlace', False)
            kwargs['comment'] = self.kwargs.get('comment', 'Animated GIF')
            if self.delay < 20:
                print(f' [GIF] Does not support delay < 20 ms (provided: {self.delay})')
            if not isinstance(kwargs['comment'], bytes):
                kwargs['comment'] = kwargs['comment'].encode('US-ASCII', errors = 'replace')
        elif format in ('tif', 'tiff',):
            # PIL.TiffImagePlugin.COMPRESSION_INFO
            # https://en.wikipedia.org/wiki/TIFF
            # 'raw' - no compression
            # 'jpeg' - with loss
            kwargs['compression'] = self.kwargs.get('compression', 'tiff_lzw')
            kwargs['format'] = 'TIFF'
        elif format in ('webp', ):
            # method is isn the range 0...6.  6 is REALLY slow
            kwargs['quality'] = self.kwargs.get('quality', 100)
            kwargs['minimize_size'] = self.kwargs.get('', True)
            kwargs['lossless'] = self.kwargs.get('lossless', True)
            kwargs['method'] = self.kwargs.get('method', 2)
        else:
            raise AttributeError(f'Format {format} not supprted.')

        if not self.collect:
            kwargs['append_images'] = iter(self.frames.get, None)
            self.proc = Thread(target = frame0.save, args = args, kwargs = kwargs)
            self.proc.start()
        else:
            self.proc = frame0
            kwargs['append_images'] = self.frames
            self.proc.save(*args, **kwargs)

    def write_frame(self, frame):
        if self.proc is self.closed:
            print(f'[{self.__class__.__name__}] Aleady closed.')
            return
        frame = fromarray(frame)
        if self.collect:
            self.frames.append(frame)
            return
        if self.proc is None:
            self.start_writer(frame)
        else:
            self.frames.put(frame)
        super().write_frame(frame)

    def get_qsize(self):
        if collect:
            return len(self.frames())
        return self.frames.qsize()

    def close(self):
        if self.proc is self.closed:
            print(f'[{self.__class__.__name__}] Aleady closed.')
            return
        if self.collect:
            self.nframes = len(self.frames)
            frame0 = self.frames.pop(0)
            print(f'[{self.__class__.__name__}] Writing buffered {self.nframes} frames.')
            self.start_writer(frame0)
        elif self.proc is None:
            print(f'[{self.__class__.__name__}] Never started.')
        else:
            self.frames.put(None)
            print(f'[{self.__class__.__name__}] Waiting for thread...')
            self.proc.join()
        self.proc = self.closed
        super().close()
        print(f'[{self.__class__.__name__}] Done.')

#-----------------------------------------------------------------------
# as backup - Matplotlib-based FFMPEG encoder

class PhonyMPLFigure(object):
    def __init__(self, size, dpi):
        self.dpi = dpi
        self.size = size
        self.size_inches = asarray(self.size) / self.dpi

    def get_size_inches(self):
        return self.size_inches

    def set_size_inches(self, *args, **kwargs):
        # maybe need to adjust dpi
        pass

class BufferMPLFFMpegWriter(FFMpegWriter):
    """
    mixin class to replace figures by frames from array

    see
    ~/Python/lib/python3.7/site-packages/matplotlib/animation.py
    """
    def setup(self, filename, frame_size, dpi):
        super().setup(PhonyMPLFigure(frame_size, dpi), filename, dpi)

    def write_frame(self, frame):
        f = self._frame_sink()
        f.write(frame)

class FFMPEGEncoderMPL(FileEncoder, FFMPEGDefaults):
    def __init__(
            self,
            filename = 'xxx.mp4',
            format = None,
            delay = 1/60,
            dpi = 100,
            **kwargs,
            ):
        super().__init__(filename)
        self.format = format
        self.fps = int(1/delay)
        self.kwargs = kwargs
        self.dpi = dpi
        self.proc = None

    def write_frame(self, frame):
        if self.proc is None:
            frame_size = frame.shape[1::-1]
            self.start_writer(frame_size)
        self.proc.write_frame(frame)
        super().write_frame(frame)

    def start_writer(self, frame_size):
        format = self.format
        if format is None:
            format = self.filename.rsplit('.', 1)[-1]
        codec, extra_args = self.defaults(format, **self.kwargs)

        self.proc = BufferMPLFFMpegWriter(
            fps = self.fps,
            codec = codec,
            extra_args = extra_args,
            )
        self.proc.setup(
            self.filename,
            frame_size,
            self.dpi)

    def close(self):
        if self.proc is None:
            print(f'[{self.__class__.__name__}] Never opened.')
            return
        if self.proc is self.closed:
            print(f'[{self.__class__.__name__}] Aleady closed.')
            return
        self.proc.close()
        self.proc = self.closed
        super().close()
        print(f'[{self.__class__.__name__}] Done.')
