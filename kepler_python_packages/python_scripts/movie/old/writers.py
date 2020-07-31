class MovieWriter(BaseMovieWriter):
    """
    stream movies to background encoder.

    can stream to several encoders in parallel

    can be used a content manager or iterable

    can use `getter` function passed as keyward argument rather than
    passing the frame on call to `write()`.  This is useful in
    connection with NoMovieWriter as dummy movie writer where you
    would not want to generate the frame and then discard it.

    getter_kw allows to pass or pre-set extra kw to getter

    TODO - non-parallel version for debugging
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
        self.parallel = kwargs.setdefault('parallel', True)
        if not hasattr(self, 'queue'):
            self.queue = kwargs.pop('queue', None)
        if self.queue is None:
            self.queue = multiprocessing.JoinableQueue()
        if not hasattr(self, 'writer'):
            self.writer = kwargs.pop('writer', None)
        if self.writer is None:
            self.writer = EncoderClient(*args, **kwargs)
            self.writer.write(self.queue)

    def write(self, *args, **kwargs):
        # Here we may add a lot more clever options for processing of
        # arguments
        if self.queue is None:
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
        self.inframeno += 1
        if self.processor is not None:
            frame = self.processor(frame, self.inframeno)
        if self.manager is not None:
            frames = self.manager(frame)
        else:
            frames = (frame, )
        for f in frames:
            self.raw_write(f)

    def raw_write(self, frame):
        if isinstance(frame, tuple):
            frame, self.outframeno = frame
        if self.finisher is not None:
            frame = self.finisher(frame, self.outframeno)
        self._raw_write(frame)
        self.outframeno += 1
    def _raw_write(self, frame):
        self.queue.put(frame)

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
                print(f'[MovieWriter] finishing {len(frames)} frames from manager(s).')
            for f in frames:
                self.raw_write(f)

    def close(self):
        if self.queue is None:
            raise Exception('Already closed.')
        self.close_manager()
        print(f'[MovieWriter] finishing {self.queue.qsize()} frames from queue.')
        self._raw_write(None)
        self.writer.close()
        self.queue = None
