#! /usr/bin/env python3
"""
Module to load and show KEPLER log files.

Can also be used as script.
"""

import logging
import datetime
import os.path
import sys
import numpy as np

import uuid

from fortranfile import FortranReader
from human import byte2human
from human import time2human
from logged import Logged
from utils import CachedAttribute
from loader import loader, _loader
from keputils import  MissingModels

log_name=['aux', 'tty', 'cmd', 'gen', 'gbn', 'exe', 'dbg', 'ali', 'lnk', 'ver']

UUID_NBYTES = 16

class LogData(Logged):
    """
    KEPLER log data interface.
    """
    _extension = 'log'
    def __init__(self,
                 filename = None,
                 silent = False,
                 **kwargs):
        """
        Constructor - provide name of dump file.
        """
        self.setup_logger(silent)
        self.file = FortranReader(filename)
        self.load_records(**kwargs)
        self.file.close()
        self.close_logger()

    def load_records(self,
                     lastmodel = sys.maxsize,
                     raise_exceptions = True,
                     ):
        self.logger_file_info(self.file)
        """Load the data records."""
        start_time = datetime.datetime.now()
        self.data=[]
        while not self.file.eof():
            try:
                record=LogRecord(self.file)
            except:
                self.logger.critical('Error in file in record {:d} (byte {:d}).  ABOARDING.'.format(
                    self.file.rpos, self.file.fpos - self.file.reclen + self.file.pos))
                if raise_exceptions:
                    raise
                break
            self.data.append(record)
            if record.ncyc >= lastmodel:
                break

        ncyc = np.array([d.ncyc for d in self.data], dtype=np.int64)
        u = np.unique(ncyc[::-1])
        ncyc_min = u[0]
        ncyc_max = u[-1]
        if len(u) != ncyc_max - ncyc_min + 1:
            jj, = np.where(np.not_equal(u[1:], u[:-1]+1))
            missing = []
            for j in jj:
                missing += [x for x in range(u[j]+1, u[j+1])]
            self.logger.error('ERROR: Missing models: ' +
                              ', '.join([str(x) for x in missing]))
            if raise_exceptions:
                raise MissingModels(models = missing,
                                    filename = self.file.filename)

        end_time = datetime.datetime.now()
        load_time = end_time - start_time
        self.nmodels = len(self.data)
        self.logger.info('version {:>9n}'.format(self.data[0].nvers))
        self.logger.info('first model red {:>9d}'.format(self.data[ 0].ncyc))
        self.logger.info(' last model red {:>9d}'.format(self.data[-1].ncyc))
        self.logger.info(str(self.nmodels)+' models loaded in '+time2human(load_time.total_seconds()))

    def show(self,
             events = None,
             all = False,
             UUID = True):
        """
        Show events.


        PARAMETERS:
            events: [(all)]
                'aux','tty','cmd','gen','gbn','exe','dbg','ali','lnk','ver'
            all: [False]
                show all records, including empty
            UUID: [True]
                show UUIDs
        """
        if not events:
            events = log_name
        for r in self.data:
            r.show(events=events, all=all, UUID=UUID)

    def cmd(self, outfile = None):
        """
        Try to reconstruct cmd file

        Write result to outfile if present, to screen outherwise.
        """
        if outfile:
            fout = open(os.path.expandvars(os.path.expanduser(outfile)),'w')
        else:
            fout = sys.stdout
        for r in self.data:
            r.cmd(outfile=fout)
        if fout != sys.stdout:
            fout.close()

    @CachedAttribute
    def ncyc(self):
        """
        return model numbers
        """
        return np.array([d.ncyc for d in self.data])


load = loader(LogData, 'loadlog')
_load = _loader(LogData, 'loadlog')
loadlog = load

class LogRecord(object):
    def __init__(self,
                 file,
                 data = True):
        "iniatialize record"
        self.file = file
        self.load(data)

    def load(self,
             data = True):
        "Load record from file"

        # TODO - add UUIDEXEC to file

        f = self.file
        f.load()
        self.nvers = f.get_i4()
        self.ncyc = f.get_i4()
        if data:
            self.uuidrun, self.uuidprog, self.uuidcycle\
                = f.get_buf(3, length = UUID_NBYTES)
            n = f.get_i4()
            self.nlog = n
            if n > 0:
                self.ilog = f.get_i4(n)
                l = f.get_i4(n)
                self.llog = l
                self.clog = f.get_sln(l)
            f.assert_eor()

    def show(self,
             events = log_name,
             all = False,
             UUID = True):
        """
        Show entries for record.
        """
        show = all
        if self.nlog > 0:
            for i,c in zip(self.ilog,self.clog):
                if log_name[i] in events:
                    show = True
                    print(r' {:<6s} {:s}'.format('['+log_name[i]+']',c))
            if show:
                if UUID:
                    print(r' CYCLE {:>8d} {!s} {!s} {!s} '.format(
                            int(self.ncyc),
                            uuid.UUID(bytes=self.uuidrun),
                            uuid.UUID(bytes=self.uuidprog),
                            uuid.UUID(bytes=self.uuidcycle)))
                else:
                    print(r' CYCLE {:>8d}'.format(int(self.ncyc)))

    def cmd(self, outfile = None):
        """
        Try to reconstruct cmd file

        Currently supported:
            p <X> <Y>
            cutsurf
        """
        if self.nlog > 0:
            out = []
            for i,c in zip(self.ilog,self.clog):
                command_list = c.split(',')
                for cmd in command_list:
                    # TTY
                    if i == 1:
                        token = cmd.split()
                        if token[0] == 'p':
                            if len(token) >= 3:
                                out.append(cmd)
                        elif token[0] == 'cutsurf':
                            out.append(cmd)

            if len(out) > 0:
                out.insert(0,'@ncyc=={}'.format(self.ncyc - 1))
                line = '\n'.join(out)+'\n'
                outfile.write(line)

    def gen(self,
            outfile = None):
        pass

    def genburn(self,
            outfile = None):
        # might be possible to get name of genburn file from generator command
        pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='KEPLER log file utility.')
    parser.add_argument('infile', nargs=1, action='store', help='input log file')

    subparser = parser.add_subparsers(dest='mode', help='possible actions', )

    # create the parser for the "cmd" command
    parser_cmd = subparser.add_parser('cmd', help='make command file')
    parser_cmd.add_argument('outfile',  nargs='?', action='store', help='output file')

    # create the parser for the "list" command
    parser_lst = subparser.add_parser('list', help='list log file content')
    parser_lst.add_argument('-u','--UUID', action='store_true', help="output UUIDs")

    group_lst = parser_lst.add_mutually_exclusive_group(required=True)
    group_lst.add_argument('-a','--all', action='store_true', help="output all log entries")
    group_lst.add_argument('-s','--selection', choices=log_name, nargs='*', action='store', help="selection ouf log file entries")

    args = parser.parse_args()
#    print(args)

    l = LogData(args.infile[0])
    if args.mode == 'list':
        l.show(args.selection, all=args.all, UUID=args.UUID)
    elif args.mode == 'cmd':
        l.cmd(args.outfile)
