"""
An adaptation of h5py for reading and writing SE files

SE files are just HDF5 files with particular conventions to support
simple reading and writing of data from 1-D stellar evolution and
nucleosynthesis codes.  This module provides a Python wrapper for h5py
that implements the conventions and restrictions expressed in the SE
library.
"""

import h5py
import numpy

class seFile(h5py.File):
    """
    Represents an SE file.
    """

    compression_level = 6

    def __init__(self, name, mode=None, driver=None, **keywords):
        """
        Initialize the seFile object.

        This just sets the file-level SE_VERSION attribute for new
        files.
        """

        h5py.File.__init__(self, name, mode, driver, **keywords)

        if len(self) == 0:
            self.attrs['SE_VERSION'] = '1.2'

    def version(self):
        """
        Return the SE version string associated with the seFile object.
        """

        try:
            ver = self.attrs['SE_VERSION']
        except KeyError:
            ver = '1.0'

        return ver

    def read(self, cyclenumber, namelist):
        """
        Read compound dataset members.

        Read some subset of the columns (structure members) from a SE
        file, and return them as a list of numpy arrays.
        """

        cyclename = namefromcycle(cyclenumber, self.version())

        if self.version() == '1.0':
            group = self
            datasetname = cyclename
        else:
            group = self[cyclename]
            datasetname = 'SE_DATASET'

        if isinstance(namelist, list):
            return [group[datasetname][n] for n in namelist]
        else:
            return group[datasetname][namelist]

    def readattr(self, cyclenumber, name):
        """
        Read an attribute from an SE cycle.

        Returns the attribute value.  Pass -1 as cyclenumber to read
        global (i.e., file-level) attributes.
        """

        if cyclenumber == -1:
            attr = self.attrs[name]
        else:
            attr = self[namefromcycle(cyclenumber, self.version())].attrs[name]

        return attr

    def readarrayattr(self, cyclenumber, tablename, colname='data'):
        """
        Read an array attribute from an SE cycle.

        Pass -1 as cyclenumber to read global (i.e., file-level)
        attributes.

        This command can also be used to read a single column of a table.

        (Added March 2010: I'm not sure why this method was designed to
        have the ability to read a single column from any data table.
        It's true that, underneath, SE "array attributes" are just tables
        with a table name other than "SE_DATASET".  In the SE library,
        "cycle" reads and writes and "array attribute" reads and writes
        are implemented with the same functions, and the same thing could
        be done here.  But exposing the colname attribute in this method
        requires users to know that array attributes store their data in a
        column named "data", which is completely an implementation detail.

        On the other hand, there are many things that I would change about
        this module now, with the benefit of hindsight, if people weren't
        already using it.  So it stays.  But I can default it to "data"!
        Thanks Python.)
        """

        if cyclenumber == -1:
            group = self
        else:
            cname = namefromcycle(cyclenumber, self.version())
            group = self[cname]

        return group[tablename][colname]

    def write(self, cyclenumber, arrays):
        """
        Write some records to a SE file.

        Arrays to write should be specified as (name, data) pairs in a
        dictionary.  The 'data' can be literal numpy arrays or Python
        lists (or really anything that numpy.asarray can reinterpret
        as an array).

        TODO: write 2D arrays as one 1D slice per cycle
        """

        cyclename = namefromcycle(cyclenumber, self.version())

        if self.version() == '1.0':
            group = self
            datasetname = cyclename
        else:
            group = self.create_group(cyclename)
            datasetname = 'SE_DATASET'

        types = []
        for name, data in arrays.items():
            if len(numpy.asarray(data).shape) > 2:
                raise ValueError("%s has dimensionality > 2" % name)

            types.append( (name,
                          numpy.asarray(data).dtype,
                          numpy.asarray(data[0]).shape) )

        dset = []
        for i in range(min(len(arrays[n]) for n in arrays)):
            dset.append(tuple(arrays[n][i] for n in arrays))

        group.create_dataset(datasetname,
                             data=numpy.asarray(dset, dtype=types),
                             compression='gzip',
                             compression_opts=self.compression_level)

    def writeattr(self, cyclenumber, name, value):
        """
        Attach an attribute to an SE cycle.

        Pass -1 as cyclenumber to write global (i.e., file-level)
        attributes.  Note that, unlike the main SE library, this
        implementation requires the cycle to exist before attaching
        attributes to it.  TODO: fix that.
        """

        if cyclenumber == -1:
            group = self
        else:
            cname = namefromcycle(cyclenumber, self.version())
            group = self[cname]

        if isinstance(value, str):
            value = numpy.array([value], dtype = 'S')

        group.attrs[name] = value

    def writearrayattr(self, cyclenumber, name, value):

        """Attach an array attribute to an SE cycle.

        Pass -1 as cyclenumber to write global (i.e., file-level)
        attributes.
        """

        ver = self.version()

        if ver == '1.0':
            raise Exception("v1.0 files don't support array attributes.")

        if cyclenumber == -1:
            group = self
        else:
            cname = namefromcycle(cyclenumber, ver)
            group = self[cname]

        dt = numpy.dtype([('data', numpy.asarray(value).dtype)])

        group.create_dataset(name,
                             data=numpy.asarray(value, dtype=dt),
                             # data=numpy.asarray([i for i
                             #                     in numpy.asarray(value)],
                             #                    dtype=dt),
                             compression='gzip',
                             compression_opts = self.compression_level)

def namefromcycle(cyclenumber, version='1.2'):
    """
    Return the dataset name corresponding to the given cycle number.

    SE 1.0 and 1.1 used dashes; SE 1.2 uses zero-padded integers with
    10-digit field width.
    """

    if version == '1.2':
        return 'cycle{:010d}'.format(cyclenumber)
    else:
        return 'cycle-' + str(cyclenumber)


def test():
    import h5py
    f = h5py.File('/home/alex/Downloads/M25.0Z2.0e-02.Ma.0023601.restart.h5','r')
    fk = h5py.File('/home/alex/Downloads/M000015Z0000.02.kepler.0000001.se.h5','r')
    fx = h5py.File('/home/alex/Downloads/M000015Z0000.02.kepler.0000000.se.h5','r')
    f.attrs['codev']
    fk.attrs['codev']
    fx.attrs['codev']
