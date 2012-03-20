__author__ = 'Samy Vilar'
__date__ = 'Mar 15, 2012'

import numpy
import numpy.ctypeslib
import ctypes

import os.path
import inspect
import gc
gc.enable()


_liblookuptable = numpy.ctypeslib.load_library('liblut', os.path.dirname(inspect.getfile(inspect.currentframe())))

float_3d_array = numpy.ctypeslib.ndpointer(dtype = numpy.float32, ndim = 3, flags='CONTIGUOUS')
int_2d_array   = numpy.ctypeslib.ndpointer(dtype = numpy.intc,  ndim = 2, flags='CONTIGUOUS')


_liblookuptable.lookuptable.argtypes = [int_2d_array,
                                        ctypes.c_uint,
                                        ctypes.c_uint,
                                        float_3d_array,
                                        float_3d_array,
                                        ctypes.c_uint]

def build_lookuptable(kwvalues):
    data, size = kwvalues['data'], kwvalues['size']
    assert data != None and size != None
    lut = lookuptable()
    lut.build(data, size)
    return lut

class lookuptable(object):
    def __init__(self, table = None):
        self.table = table
        self._sums = None
        self._counts = None

    @property
    def table(self):
        return self._table
    @table.setter
    def table(self, value):
        self._table = value

    @property
    def size(self):
        return self._size
    @size.setter
    def size(self, value):
        self._size = value

    @property
    def sums(self):
        return self._sums
    @sums.setter
    def sums(self, values):
        self._sums = values

    @property
    def counts(self):
        return self._counts
    @counts.setter
    def counts(self, values):
        self._counts = values

    def build(self, data = None, size = None):
        assert data != None and size != None
        self.size = size
        counts = numpy.zeros((size, size, size), dtype = 'float32')
        sums = numpy.zeros((size, size, size), dtype = 'float32')

        values = (numpy.round((data * self.size - 0.5))).astype('intc') # redefine values to be used properly as indices
        values[values >= self.size] = self.size - 1 # make sure none of the values exceed max, if they do simply set them to the max.
        values[values < 0] = 0


        shape = numpy.asarray(data.shape, dtype = 'uintc')
        _liblookuptable.lookuptable(values,
                                    shape[0],
                                    shape[1],
                                    sums,
                                    counts,
                                    numpy.asarray([size,], dtype = 'uintc')[0])

        self.counts = counts
        self.sums = cums
        self.table = numpy.zeros((size, size, size), dtype = 'float32')
        non_zero_locations = counts != 0
        self.table[non_zero_locations] = sums[non_zero_locations]/counts[non_zero_locations]

        del sums
        del non_zero_locations







