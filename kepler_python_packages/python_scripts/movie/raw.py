import numpy as np
import gzip
import bz2
import lzma
import os.path
import multiprocessing

# TODO - add converter routine (e.g., writing gz is a lot faster than bz and xz)

class RawMovie(object):
    compression_modes = ('gz', 'bz', 'xz')
    def __init__(self, filename):
        if filename.endswith(('.raw.gz', '.raw.bz', '.raw.xz')):
            self.filecompression = filename[-2:]
            self.framecompression = 'raw'
        else:
            self.filecompression = 'raw'
            if filename.endswith(('.rgz', '.rbz', '.rxz')):
                self.framecompression = filename[-2:]
            else:
                assert filename.endswith('.raw')
                self.framecompression = 'raw'
        self.filename = os.path.expandvars(os.path.expanduser(filename))


# todo - add full Frame data:
#        allow joing of partical files, sorting, appending, etc.

# todo - add seek, build directory, buffer (compressed, uncompressed), etc.
# todo - add EOF
# add seek (for r?z)

# todo - add buffering (load entire file, readahead, buffer recent
#        loaded frames, keep director of record locations for r?z formats

class RawMovieReader(RawMovie):
    def __init__(self, filename, **kwargs):
        # kwargs currently ignored
        super().__init__(filename)
        if self.filecompression == 'gz':
            self.file = gzip.open(filename,'rb')
        elif self.filecompression == 'bz':
            self.file = bz2.BZ2File(filename,'rb')
        elif self.filecompression == 'xz':
            self.file = lzma.LZMAFile(self.filename,'rb')
        else:
            self.file = open(filename,'rb',-1)
        if self.framecompression == 'gz':
            self.decompress = gzip.decompress
        elif self.framecompression == 'bz':
            self.decompress = bz2.decompress
        elif self.framecompression == 'xz':
            self.decompress = lzma.decompress
        else:
            self.decompress = lambda x: x
        self._read_header()
        self.pos = 0

    def _read_header(self):
        data = self.file.read(16)
        self.version = np.ndarray((), dtype = np.int64, buffer = data[0:8])
        compression = data[8:].decode().strip('\x00')
        assert self.framecompression == compression

    def read(self):
        data = self.file.read(32)
        if len(data) < 32:
            raise EOFError()
        values = np.ndarray((4), dtype=np.int64, buffer=data)
        dims = tuple(values[0:3].tolist())
        size = values[3].tolist()
        data = self.file.read(size)
        if len(data) < size:
            raise EOFError()
        data = self.decompress(data)
        frame = np.ndarray(dims, dtype=np.uint8, buffer = data)
        self.pos += 1
        return frame

    def __iter__(self):
        return self
    def __next__(self):
        try:
            return self.read()
        except EOFError as e:
            print(f'[RawReader] Encountered {e!r}')
            raise StopIteration()

    def __enter__(self, *args, **kwargs):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    def close(self):
        self.file.close()

# todo - add 'append' mode
#      - add queue write mode as process
#      - add server write mode
class RawMovieWriter(RawMovie):
    version = 10000
    def __init__(self, filename, **kwargs):
        # kwargs currently ignored.  Should be checked.
        super().__init__(filename)
        if self.filecompression == 'gz':
            self.file = gzip.open(filename,'wb')
        elif self.filecompression == 'bz':
            self.file = bz2.BZ2File(filename,'wb')
        elif self.filecompression == 'xz':
            self.file = lzma.LZMAFile(self.filename,'wb')
        else:
            self.file = open(filename,'wb',-1)
        if self.framecompression == 'gz':
            self.compress = gzip.compress
        elif self.framecompression == 'bz':
            self.compress = bz2.compress
        elif self.framecompression == 'xz':
            self.compress = lzma.compress
        else:
            self.compress = lambda x: x
        self._write_header()
        self.pos = 0

        if self.framecompression != 'raw':
            self.parallel = kwargs.get('parallel', True)
        else:
            self.parallel = False
        if self.parallel:
            self.nparallel = kwargs.get('nparallel', None)
            if self.nparallel is None:
                self.nparallel = multiprocessing.cpu_count()
            self.queue_r = multiprocessing.JoinableQueue()
            self.queue_z = multiprocessing.JoinableQueue()
            self.processes = list()
            for i in range(self.nparallel):
                self.processes.append(
                    multiprocessing.Process(
                        target = self._compress,
                        args = (self.queue_r, self.queue_z, self.compress),
                        )
                    )
                self.processes[i].start()
            self.writer = multiprocessing.Process(
                target = self._writer,
                args = (self.queue_z, self.nparallel, self.file, self._write),
                )
            self.writer.start()

    def _write_header(self):
        version = np.array(self.version, dtype = np.int64)
        x = bytearray(8)
        x[0:len(self.framecompression)] = self.framecompression.encode()
        self.file.write(version.data.tobytes() + x)

    @staticmethod
    def _compress(queue_r, queue_z, compress):
        while True:
            items = queue_r.get()
            if len(items) > 1:
                dims, data, pos = items
                data = compress(data)
                items = dims, data, pos
            queue_z.put(items)
            queue_r.task_done()
            if len(items) == 1:
                break
        print('[RawWriter] Done _compress process.')

    @staticmethod
    def _writer(queue_z, nparallel, file, write):
        next = 0
        buf = dict()
        while True:
            items = queue_z.get()
            if len(items) > 1:
                dims, data, pos = items
                buf[pos] = (dims, data)
            while True:
                values = buf.pop(next, None)
                if values is not None:
                    write(file, *values)
                    next += 1
                    queue_z.task_done()
                else:
                    break
            if len(buf) > 0:
                print(f'[RawWriter] _writer buffering {len(buf)} frames.')
            if len(items) == 1:
                nparallel -= 1
                queue_z.task_done()
            if nparallel == 0:
                assert len(buf) == 0
                break
        print('[RawWriter] Done _writer process.')

    @staticmethod
    def _write(file, dims, data):
        size = np.array(len(data), dtype=np.int64)
        size = size.data.tobytes()
        file.write(dims + size)
        file.write(data)

    def write(self, frame):
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 4
        assert frame.nbytes == np.product(frame.shape)
        assert frame.dtype == np.uint8
        dims = np.array(frame.shape, dtype=np.int64)
        dims = dims.data.tobytes()
        data = frame.data.tobytes()
        if self.parallel:
            self.queue_r.put((dims, data, self.pos))
            self.pos += 1
            return
        data = self.compress(data)
        self._write(self.file, dims, data)
        self.pos += 1

    def close(self):
        if self.parallel:
            for i in range(self.nparallel):
                self.queue_r.put((self.pos,))
            print('[RawWriter] Waiting for writer queue')
            self.queue_r.join()
            print('[RawWriter] Waiting for _compress processes')
            for p in self.processes:
                p.join()
            print('[RawWriter] Waiting for writer process')
            self.writer.join()

        self.file.close()
    def __enter__(self, *args, **kwargs):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

class MemoryMovie(object):
    def __init__(self, retain = True, compression = 'gz', **kwargs):
        self.buffer = list()
        super().__init__(*args, **kwargs)
        if compression is None:
            self.compress = lambda x: copy(x)
            self.decompress = self.compress
        elif compression in ('gz', 'bz', 'xz', ):
            if compression == 'gz':
                compress = gzip.compress
                decompress = gzip.decompress
            elif compression == 'bz':
                compress = bz2.compress
                decompress = bz2.decompress
            elif compression == 'xz':
                compress = lzma.compress
                decompress = lzma.decompress
            else:
                raise Exception('MemoryMovie: Unknown compression')
            self.compress = lambda x: compress(pickle.dumps(x))
            self.decompress = lambda x: pickle.loads(decompress(x))
        else:
            raise Exception('MemoryMovie: Unknown compression')
    def write(self, frame):
        self.buffer.append(self.compress(frame))
    def reset(self, start = 0):
        self.count = start
    def getter(self):
        if not retain:
            frame = self.buffer.pop(0, None)
        else:
            frame = self.buffer.get(self.count, None)
        if frame is not None:
            self.count += 1
            frame = self.decompress(frame)
        return frame
    def __getitem__(self, key):
        return self.buffer[key]
    def __iter__(self):
        return iter(self.buffer)
    def close(self):
        pass
