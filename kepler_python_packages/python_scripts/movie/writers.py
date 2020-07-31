
import multiprocessing
import multiprocessing.pool
import contextlib
import time
import lzma
import gzip
import bz2
import pickle
import os

from abc import ABC, abstractmethod
from copy import copy

from collections.abc import Iterable

from human import time2human

from .encoders import ADD_WRITER, DEL_WRITER
from .managers import BaseMovieManager
from .encoders import Encoder, EncoderClient, QueueFrameSorter
from .frames import Frame

#=======================================================================
# Movie Writer

def make(mp, *args, **kwargs):
    if mp in (False, 'serial', ):
        MovieWriter.make(*args, **kwargs)
    elif mp in (True, 'parallel', ):
        ParallelMovie.make(*args, **kwargs)
    elif mp in ('pool', ):
        PoolMovie.make(*args, **kwargs)
    elif mp in (None, ):
        NoMovieWriter.make(*args, **kwargs)
    else:
        raise AttributeError('Unknown Movie Mode.')

class BaseMovieWriter(object):
    closed = object()
    def __init__(self, *args, **kwargs):
        self.batch = kwargs.pop('batch', False)
        self.starttime = time.time()
    def write(self, *args, **kwargs):
        raise NotImplementedError()
    def raw_write(self, *args, **kwargs):
        raise NotImplementedError()
    def get_batch(self):
        return self.batch
    def set_batch(self, batch = True):
        self.batch = batch
    def close(self):
        # need to ensure manager.close is called here as well.
        raise NotImplementedError()
    @contextlib.contextmanager
    def writing(self, manager = BaseMovieManager()):
        # need to be able to use exisiting managers, as chain
        old_manager = self.set_manager(manager)
        try:
            yield self
        finally:
            self.set_manager(old_manager)
            try:
                for f in manager.close():
                    self.raw_write(f)
            except:
                pass
            if not self.batch:
                self.close()
    # provide iterator functions
    def iframes(self, *args, manager = BaseMovieManager()):
        """
        Takes slice parameter stop or (start, stop [,step])

        Iterates quasi-indefinetively if no limits are provided.
        """
        if len(args) == 0:
            s = slice(None)
        elif isinstance(args[0], slice):
            s = args[0]
            assert len(args) == 1
        else:
            s = slice(*args)
        with self.writing(manager = manager) as w:
            for i in range(*s.indices(2**32-1)):
                yield i
                w.write()
    def range(self, *args, manager = BaseMovieManager()):
        """
        Just iterate of range, simpler version of the above)
        """
        with self.writing(manager = manager) as w:
            for item in range(*args):
                yield item
                w.write()
    def iter(self, it, manager = BaseMovieManager()):
        """
        Iterates frames over values of provided iterable
        """
        with self.writing(manager = manager) as w:
            for item in it:
                yield item
                w.write()
    def enumerate(self, it, manager = BaseMovieManager()):
        """
        Iterates frames by enumerating provided iterable
        """
        with self.writing(manager = manager) as w:
            for i,item in enumerate(it):
                yield i,item
                w.write()

class MovieWriter(BaseMovieWriter):
    """
    stream movies to background encoder.

    can stream to several encoders in parallel

    can be used a content manager or iterable

    can use `getter` function passed as keyward argument rather than
    passing the frame on call to `write()`.  This is useful in
    connection with NoMovieEncoder as dummy movie encoder where you
    would not want to generate the frame and then discard it.

    getter_kw allows to pass or pre-set extra kw to getter

    parallel=False:
        Non-parallel version for debugging
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.getter = kwargs.pop('getter', None)
        self.getter_kw = kwargs.pop('getter_kw', None)
        self.processor = kwargs.pop('processor', None)
        self.manager = kwargs.pop('manager', None)
        self.finisher = kwargs.pop('finisher', None)
        self.inframeno = kwargs.pop('inframeno', 0)
        self.outframeno = kwargs.pop('outframeno', 0)
        self.parallel = kwargs.pop('parallel', True)
        if not hasattr(self, 'encoder'):
            self.encoder = kwargs.pop('encoder', None)
        if self.encoder is None:
            if self.parallel:
                enc = EncoderClient
            else:
                enc = Encoder
            self.encoder = enc(*args, parallel = self.parallel, **kwargs)
        self.starttime = time.time()
        self.nframes = 0

    def write(self, *args, **kwargs):
        # Here we may add a lot more clever options for processing of
        # arguments
        if self.encoder is None:
            raise Exception('Already closed.')
        getter_kw = kwargs.pop('getter_kw', self.getter_kw)
        if getter_kw is None:
            getter_kw = dict()
        if len(args) == 0 and len(kwargs) == 0 and self.getter is not None:
            frame = self.getter(**getter_kw)
        elif len(args) == 1:
            frame = args[0]
            if callable(frame):
                frame = frame()
        elif len(kwargs) > 0:
            frame = kwargs.get('frame', None)
            if frame is None:
                getter = kwargs.get('getter', None)
                if callable(getter):
                    frame = getter(**getter_kw)
        else:
            frame = None
        if frame is None:
            raise AttributeError('Could not get frame.')
        assert isinstance(frame, Frame)
        if frame.iframe is None:
            frame.iframe = self.inframeno
        self.inframeno += 1
        if self.processor is not None:
            frame = self.processor(frame)
        if self.manager is not None:
            frames = self.manager(frame)
        else:
            frames = (frame, )
        for f in frames:
            self.raw_write(f)

    def raw_write(self, frame):
        assert isinstance(frame, Frame)
        if frame.oframe is None:
            frame.oframe = self.outframeno
        if self.finisher is not None:
            frame = self.finisher(frame)
        self._raw_write(frame)
        self.outframeno += 1
    def _raw_write(self, frame):
        self.nframes += 1
        self.encoder.write_frame(frame)

    def get_manager(self):
        return self.manager
    def set_manager(self, manager):
        old_manager, self.manager = self.manager, manager
        return old_manager
    def clear_manager(self):
        return self.set_manager(None)

    def set_processor(self, processor):
        old_processor, self.processor = self.processor, processor
        return old_procssor
    def get_processor(self):
        return self.processor
    def clear_processor(self):
        return self.set_processor(None)

    def set_finisher(self, finisher):
        old_finisher, self.finisher = self.finisher, finisher
        return old_finisher
    def get_finisher(self):
        return self.finisher
    def clear_finisher(self):
        return self.set_finisher(None)

    def clear(self):
        return self.clear_processor(), self.clear_manager(), self.clear_finisher()
    def reset(self, inframeno = True, outframeno = True):
        result = []
        if inframeno:
            result.append(self.inframeno)
            self.inframeno = 0
        if outframeno:
            result.append(self.outframeno)
            self.outframeno = 0
        return tuple(result)

    def set_getter(self, getter):
        self.getter = getter
    def set_getter_kw(self, getter_kw):
        self.getter_kw = getter_kw

    def close_manager(self):
        if self.manager is not None:
            frames = self.manager.close()
            if len(frames) > 0:
                print(f'[{self.__class__.__name__}] finishing {len(frames)} frames from manager(s).')
            for f in frames:
                self.raw_write(f)

    def close(self):
        if self.encoder is None:
            raise Exception('Already closed.')
        self.close_manager()
        runtime = time.time() - self.starttime
        print(f'[{self.__class__.__name__}] Generated {self.nframes} frames in {time2human(runtime)} ({self.nframes/runtime:7.3f} fps)')
        qsize = self.encoder.get_qsize()
        if qsize is None:
            qsize = ''
        else:
            qsize = f'{qsize} '
        print(f'[{self.__class__.__name__}] finishing {qsize}frames.')
        self.encoder.close()
        self.encoder = None

    @classmethod
    def make(
            cls,
            filename = None,
            func = None,
            canvas = None,
            values = None,
            fargs = (),
            fkwargs = dict(),
            base = None,
            bargs = (),
            bkwargs = dict(),
            cargs = (),
            ckwargs = dict(),
            enum = False,
            data = False,
            batch = False,
            writer = None,
            start = 0,
            **kwargs,
            ):
        """
        requites call signature as specified
          enum: takes iframe as first arg
          data: takes data value as next arg
          if str specify kwarg[s] instead
        plus a 'fig' kwarg
        """
        if isinstance(canvas, type):
            canvas = canvas(*cargs, **ckwargs)
        if canvas is not None:
            fig = canvas.get_canvas()
        if base is not None:
            base = base(*bargs, **bkwargs)
            func = getattr(base, func)
        if writer is None:
            writer = cls(filename, **kwargs)
        if isinstance(values, slice):
            values = range(*values.indices(values.stop))
        elif not isinstance(values, Iterable):
            values = range(start, start + values)
        for iframe, idata in enumerate(values):
            iargs = []
            ikwargs = dict()
            if isinstance(enum, str):
                ikwargs[enum] = iframe
            elif enum:
                iargs.append(iframe)
            if isinstance(data, str):
                ikwargs[data] = idata
            elif data:
                iargs.append(idata)
            if canvas is not None:
                canvas.clear()
                ikwargs['fig'] = fig
                func(*iargs, *fargs, **ikwargs, **fkwargs)
                frame = canvas.get_frame(iframe)
            else:
                frame = func(*iargs, *fargs, **ikwargs, **fkwargs)
                frame.iframe = iframe
            writer.write(frame)
        if not batch:
            writer.close()
        else:
            return writer

#========================================================================
# some proxy writers

class NoMovieWriter(BaseMovieWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = object()
    def write(self, *args, **kwargs):
        if self.encoder is None:
            raise Exception('Already closed.')
    def close(self):
        if self.encoder is None:
            raise Exception('Already closed.')
        self.encoder = None
    def set_manager(self, manager):
        pass

class _PhoneyEncoder(object):
    def __init__(self, *args, **kwargs):
        pass
    def join(self):
        pass

class _PhoneyQueue(object):
    def __init__(self, *args, **kwargs):
        pass
    def join(self):
        pass
    def put(self, frame):
        pass
    def qsize(self):
        return 0

# alternatively, use memory encoder.  currently, you'd have to use
# non-parallel version, otherwise, the copy of the object that recives
# the frames will be lost.
#
# TODO - pickle needs to send proxy object that connects back to the
#        original object by setting up connection.  see
#        https://docs.python.org/3/library/pickle.html for setting up
#        pickle behaviour.  Maybe managers or shared memeory (python
#        3.8)


class MemoryMovieWriter(MovieWriter):
    def __init__(self, retain = True, compression = 'gz', **kwargs):
        self.encoder = _PhoneyEncoder()
        self.mem = MemoryMovie(retain = retain, compression = compression)
        super().__init__(*args, **kwargs)
    def _raw_write(self, frame):
        self.mem.write(frame)
    def reset(self, start = 0):
        self.mem.reset(start)
    def getter(self):
        return self.mem.getter()
    def __getitem__(self, key):
        return self.mem[key]
    def __iter__(self):
        return self.mem.__iter__()
    # update to make usable as queue?
    # add get/put interface or queue-like interface object?


#=======================================================================
# Parallel movie writer interface

#class ParallelMovieFrames(ABC):
class ParallelMovieFrames(object):
    """
    define interface
    """
#    @abstractmethod
    def __init__(self):
        """
        set up object and (likely) 'canvas'
        """
        pass

#    @abstractmethod
    def draw(self, iframe, data = None):
        """
        take frame number and (optional) data, return Frame instance
        """
        pass

#    @abstractmethod
    def close(self):
        """
        opportunity to free resources, e.g., delete render windows
        """
        pass

class ParallelMovieFramesFunction(ParallelMovieFrames):
    def __init__(
            self, func,
            canvas = None,
            fargs = (),
            fkwargs = dict(),
            base = None,
            bargs = (),
            bkwargs = dict(),
            cargs = (),
            ckwargs = dict(),
            enum = False,
            data = False,
            ):
        self.func = func
        self.fargs = fargs
        self.fkwargs = fkwargs
        if canvas is not None:
            canvas = canvas(*cargs, **ckwargs)
        self.canvas = canvas
        self.enum = enum
        self.data = data
        if base is not None:
            self.base = base(*bargs, **bkwargs)
        else:
            self.base = None

    def draw(self, iframe, data = None):
        iargs = []
        ikwargs = dict()
        if isinstance(self.enum, str):
            ikwargs[self.enum] = iframe
        elif self.enum:
            iargs.append(iframe)
        if isinstance(self.data, str):
            ikwargs[self.data] = data
        elif self.data:
            iargs.append(data)
        if self.base is not None:
            func = getattr(self.base, self.func)
        else:
            func = self.func
        if self.canvas is not None:
            self.canvas.clear()
            fig = self.canvas.get_canvas()
            ikwargs['fig'] = fig
            func(*iargs, *self.fargs, **ikwargs, **self.fkwargs)
            frame = self.canvas.get_frame(iframe)
        else:
            frame = func(*iargs, *self.fargs, **ikwargs, **self.fkwargs)
            frame.iframe = iframe
        return frame

    def close(self):
        if self.canvas is not None:
            self.canvas.close()

DEFAULT_PARALLEL_NICE = 19

class ParallelMovieProcess(multiprocessing.Process):
    """
    Process to make frames
    """
    def __init__(self, qi, qo, qr, generator, gargs, gkwargs, tag, **kwargs):
        super().__init__()
        self.qi = qi
        self.qo = qo
        self.qr = qr
        if gargs is None:
            gargs = list()
        if gkwargs is None:
            gkwargs = dict()
        self.generator = generator(*gargs, **gkwargs)
        self.nice = kwargs.get('nice', DEFAULT_PARALLEL_NICE)
        self.tag = tag
    def run(self):
        print(f'[{self.__class__.__name__}] [{self.tag}] [{self.pid:06d}] Starting.')
        self.qr.put(self.tag)
        os.nice(self.nice)
        self.qo.put(ADD_WRITER)
        while True:
            iframe, data = self.qi.get()
            if iframe is None:
                self.qo.put(DEL_WRITER)
                self.qi.task_done()
                break
            print(f'[{self.__class__.__name__}] [{self.tag}] [{self.pid:06d}] processing frame {iframe}')
            frame = self.generator.draw(iframe, data)
            if frame.iframe is not None:
                assert frame.iframe == iframe
            else:
                frame.iframe = iframe
            self.qo.put(frame)
            self.qi.task_done()
        self.generator.close()
        print(f'[{self.__class__.__name__}] [{self.tag}] [{self.pid:06d}] Done.')

class ParallelMovie(MovieWriter):
    """
    Parallel Movie Writer using processes
    """
    def __init__(
            self,
            *args,
            generator = None,
            nparallel = None,
            gargs = tuple(),
            gkwargs = dict(),
            # config options
            **kwargs,
            ):
        super().__init__(*args, **kwargs)
        self.generator = generator
        if nparallel is None:
            nparallel = multiprocessing.cpu_count()
        self.nparallel = nparallel
        self.gargs = gargs
        self.gkwargs = gkwargs

    def run(self,
            sequence,
            generator = None,
            gargs = None,
            gkwargs = None,
            nparallel = Ellipsis,
            batch = False,
            start = 0,
            ):
        """
        sequence can be
          - iterable
          - number of items
          - slice (start is ignored)
        """
        if generator is None:
            generator = self.generator
        if gargs is None:
            gargs = self.gargs
        if gkwargs is None:
            gkwargs = self.gkwargs
        if nparallel is Ellipsis:
            nparallel = self.nparallel
        if nparallel is None:
            nparallel = multiprocessing.cpu_count()

        starttime = time.time()
        framequeue = multiprocessing.JoinableQueue()
        print(f'[{self.__class__.__name__}] starting {nparallel} processes.')
        processors = dict()
        runqueue = multiprocessing.Queue()
        squeue = QueueFrameSorter()
        inqueue = squeue.get_input_queue()
        outqueue = squeue.get_output_queue()
        for i in range(nparallel):
            tag = f'{i+1}'
            p = ParallelMovieProcess(
                    framequeue,
                    inqueue,
                    runqueue,
                    generator,
                    gargs,
                    gkwargs,
                    tag = tag,
                    )
            p.daemon = True
            p.start()
            processors[tag] = p
        if isinstance(sequence, Iterable):
            for iframe, data in enumerate(sequence, start=start):
                framequeue.put((iframe, data))
                nframes = iframe + 1
        elif isinstance(sequence, slice):
            for iframe in range(*seqeunce.indices(sequence.stop)):
                framequeue.put((iframe, None))
                nframes = iframe + 1
        else:
            for iframe in range(start, start + sequence):
                framequeue.put((iframe, None))
            nframes = sequence
        print(f'[{self.__class__.__name__}] Sent {nframes} frames to render.')
        working = []
        for _ in range(nparallel):
            try:
                tag = runqueue.get(timeout=1)
            except:
                break
            working.append(tag)
            framequeue.put((None, None))
        framequeue.close()
        print(f'[{self.__class__.__name__}] {len(working)}/{nparallel} working processes. ')
        # for i in range(self.nparallel):
        #     self.framequeue.put((None, None))
        print(f'[{self.__class__.__name__}] Writing results from processes ... ')
        mframes = 0
        while True:
            xframe = outqueue.get()
            if xframe is None:
                break
            self.write(xframe)
            mframes += 1
        print(f'[{self.__class__.__name__}] Received {mframes} frames from processors.')
        print(f'[{self.__class__.__name__}] Waiting for queues ...')
        framequeue.join()
        while outqueue.qsize() > 0:
            print(f'[debug] {outquete.get()!r}')
        outqueue.cancel_join_thread()
        squeue.close()
        inqueue.join()
        # outqueue.join()
        print(f'[{self.__class__.__name__}] Waiting for frame processors ...')
        for tag in working:
            print(f'[{self.__class__.__name__}] joining task {tag}.')
            processors.pop(tag).join()
        for tag, p in processors.items():
            print(f'[{self.__class__.__name__}] [DEBUG] killing task {tag}.')
            p.kill()
        runtime = time.time() - starttime
        print(f'[{self.__class__.__name__}] Generated {nframes} frames in {time2human(runtime)} ({nframes/runtime:7.3f} fps)')
        print(f'[{self.__class__.__name__}] Waiting for encoder ...')
        encoderstarttime = time.time()
        if not batch:
            self.close()
        encoderruntime = time.time() - encoderstarttime
        runtime = time.time() - starttime
        print(f'[{self.__class__.__name__}] Encoder done in {time2human(encoderruntime)} ({time2human(runtime)} total, {nframes/runtime:7.3f} fps).')

    @classmethod
    def make(
            cls,
            filename = None,
            func = None,
            canvas = None,
            values = None,
            fargs = (),
            fkwargs = dict(),
            base = None,
            bargs = (),
            bkwargs = dict(),
            cargs = (),
            ckwargs = dict(),
            enum = False,
            data = False,
            nparallel = None,
            batch = False,
            writer = None,
            **kwargs,
            ):
        """
        requites call signature as specified
          enum: takes iframe as first arg
          data: takes data value as next arg
          'fig' kwarg if a canvas class is provided
        """
        gkwargs = dict(
            func = func,
            canvas = canvas,
            fargs = fargs,
            fkwargs = fkwargs,
            base = base,
            bargs = bargs,
            bkwargs = bkwargs,
            cargs = cargs,
            ckwargs = ckwargs,
            enum = enum,
            data = data,
            )
        if writer is None:
            writer = cls(filename, **kwargs)
        writer.run(
            values,
            nparallel = nparallel,
            generator = ParallelMovieFramesFunction,
            gkwargs = gkwargs,
            gargs = tuple(),
            batch = batch,
            )
        if batch:
            return writer

#-----------------------------------------------------------------------
# A ProcessPool version
# add ProcessPool version? (seems less efficient)

class PoolMovieFrames(object):
    def __init__(
            self,
            generator = None,
            gargs = tuple(),
            gkwargs = dict(),
            nice = DEFAULT_PARALLEL_NICE,
            ):
        self.generator = generator
        self.gargs = gargs
        self.gkwargs = gkwargs
        self.nice = nice

    def draw(self, task):
        os.nice(self.nice)
        iframe, data = task
        generator = self.generator(*self.gargs, **self.gkwargs)
        frame = generator.draw(iframe, data)
        if frame.iframe is not None:
            assert frame.iframe == iframe
        else:
            frame.iframe = iframe
        generator.close()
        return frame

class PoolMovie(ParallelMovie):
    """
    Parallel Movie writer using a Processor Pool

    inherits from ParallelMovie but overwrites the process generation
    in the 'run' method.

    Also supports the 'make' interface.
    """
    def run(self,
            sequence,
            generator = None,
            gargs = None,
            gkwargs = None,
            nparallel = Ellipsis,
            batch = False,
            start = 0,
            maxtasksperchild = None,
            nice = DEFAULT_PARALLEL_NICE,
            ):
        """
        the generator needs to be a function takes *args = (iframe,
        data) and returns a Frame object
        """
        if generator is None:
            generator = self.generator
        if gargs is None:
            gargs = self.gargs
        if gkwargs is None:
            gkwargs = self.gkwargs
        if nparallel is Ellipsis:
            nparallel = self.nparallel
        if nparallel is None:
            nparallel = multiprocessing.cpu_count()

        starttime = time.time()
        print(f'[{self.__class__.__name__}] starting Pool with {nparallel} processes.')
        # pool = multiprocessing.pool.Pool(nparallel)
        pool = multiprocessing.pool.Pool(nparallel, None, None, maxtasksperchild) # to limit how much is processed before restarting

        if isinstance(sequence, Iterable):
            items = enumerate(sequence, start=start)
        elif isinstance(sequence, slice):
            items = zip(range(sequence), itertools.repeat(None))
        else:
            items = enumerate(itertools.repeat(None, sequence), start=start)
        # let's have imap do the sorting
        func = PoolMovieFrames(generator, gargs, gkwargs, nice=nice).draw
        iresults = pool.imap(func, items)
        nframes = 0
        for iframe, frame in enumerate(iresults, start=start):
            # maybe set frame.iframe
            frame.iframe = iframe
            self.write(frame)
            nframes += 1
        print(f'[{self.__class__.__name__}] Closing Pool.')
        pool.close()
        runtime = time.time() - starttime
        print(f'[{self.__class__.__name__}] Generated {nframes} frames in {time2human(runtime)} ({nframes/runtime:7.3f} fps)')
        print(f'[{self.__class__.__name__}] Waiting for encoder ...')
        encoderstarttime = time.time()
        if not batch:
            self.close()
        encoderruntime = time.time() - encoderstarttime
        runtime = time.time() - starttime
        print(f'[{self.__class__.__name__}] Encoder done in {time2human(encoderruntime)} ({time2human(runtime)} total, {nframes/runtime:7.3f} fps).')


#-----------------------------------------------------------------------
# Draft of single clients on independent processess
# needs to provide mechanism to generate frame numbers ...
# Setting up/getting queue through a manager and provide server

MMW_PORT = 51000
MMW_HOST = 'localhost'
MMW_AUTH = b'JimOHjoEDDheHNVo7kydp4kAArVpd+P6+rV/BiCSlkc'
MMW_CLOSE = 'MMW_CLOSE'
MMW_SET_M = 'MMW_SET_M'
MMW_SET_F = 'MMW_SET_F'
MMW_SET_P = 'MMW_SET_P'

class QueueDecoder(multiprocessing.Process):
    def __init__(self, queue, rq, wargs, wkwargs):
        self.queue = queue
        self.rq = rq
        self.writer = MovieWriter(*wargs, **wkwargs)
        self.sorter = QueueFrameSorter(
            batch = True,
            num = 'oframe',
            )
        self.qi = self.sorter.get_input_queue()
        self.qo = self.sorter.get_output_queue()

    def run(self):
        # for sorting/closing: need to know what data to expect ...

        self.no = 0
        while True:
            frame = self.queue.get()
            if frame is None:
                self.no += 1
                self.queue.task_done()
                continue
            if isinstance(frame, tuple):
                if frame[0] == MMW_CLOSE:
                    self.qi.put(ENC_CLOSE)
                elif frame[0] == MMW_SET_M:
                    self.rq.put(self.writer.set_manager(frame[1]))
                elif frame[0] == MMW_SET_F:
                    self.rq.put(self.writer.set_finisher(frame[1]))
                elif frame[0] == MMW_SET_P:
                    self.rq.put(self.writer.set_processor(frame[1]))
                else:
                    print('[DEBUG] Unknown command')
            else:
                self.qi.put(frame)
            self.queue.task_done()
            done = False
            while self.qo.qsize() > 0:
                frame = self.qo.get()
                if frame == None:
                    done = True
                    break
                frame.iframe = frame.oframe
                frame.oframe = None
                self.writer.write(frame)
            if done:
                break
        self.writer.close()
        self.sorter.close()

class MultiMovieWriterServer(object):
    def __init__(self, *args, **kwargs):
        self.host = MMW_HOST
        self.port = kwargs.pop('port', MMW_PORT)
        self.auth = kwargs.pop('auth', MMW_AUTH)
        self.framequeue = multiprocessing.JoinableQueue()
        self.resultqueue = multiprocessing.JoinableQueue()
        class QueueManager(BaseManager): pass
        QueueManager.register('get_queue', callable=lambda: self.framequeue)
        self.manager = QueueManager(
            address=(
                self.port,
                self.address),
            authkey=self.auth)
        self.manager.run()
        self.decoder = QueueDecoder(
            self.framequeue,
            self.resultqueue,
            args, kwargs)

        # add setups w/r tranlation of oframes/iframes to be done at client side

    # set up mechanism to communicate with QueueDecoder to set
    # processors and managers
    def set_manager(self, manager):
        self.framequeue.put((MMW_SET_M, manager))
        return self.resultqueue.get()
    def set_processor(self, manager):
        self.framequeue.put((MMW_SET_P, manager))
        return self.resultqueue.get()
    def set_finisher(self, manager):
        self.framequeue.put((MMW_SET_F, manager))
        return self.resultqueue.get()

    def close(self):
        self.framequeue.put((MMW_CLOSE, None))
        self.framequeue.close()
        self.resultqueue.close()
        self.manager.shutdown()

class QueueEncoder(object):
    def __init__(self, queue, offset=0):
        self.queue = queue
        self.offset = offset
    def write_frame(self, frame):
        frame.oframe += self.offset
        self.queue.put(frame)
    def close(self):
        self.queue.close() #???

class MultiMovieWriterClient(MovieWriter):
    """
    Single client for multi-movie writing
    """
    def __init__(self, *args, **kwargs):
        self.host = kwargs.get('host', MMW_HOST)
        self.port = kwargs.pop('port', MMW_PORT)
        self.auth = kwargs.pop('auth', MMW_AUTH)

        # this could also be taken care of my MovieWriter itself
        self.offset = kwargs.pop('offset', 0)

        class QueueManager(BaseManager): pass
        QueueManager.register('get_queue')
        self.manager = QueueManager(address=(self.port, self.address),
                                    authkey=self.auth)
        self.manager.connect()
        self.framequeue = self.manager.get_queue()
        assert 'encoder' not in kwargs
        kwargs['encoder'] = QueueEncoder(self.framequeue)
        super().__init__(*args, **kwargs)



#-----------------------------------------------------------------------
# TODO - add cluster MPI version
# https://rabernat.github.io/research_computing/parallel-programming-with-mpi-for-python.html
# from mpi4py import MPI

# TODO
# - raw storage / buffering to memory or disk
# - read movie as frame source
