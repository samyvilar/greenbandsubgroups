__author__ = 'Samy Vilar'
__date__ = 'Mar 15, 2012'

import numpy
import numpy.ctypeslib
import ctypes

import os.path
import inspect


_liblookuptable = numpy.ctypeslib.load_library('liblut', os.path.dirname(inspect.getfile(inspect.currentframe())))

unsigned_int_3d_array = numpy.ctypeslib.ndpointer(dtype = numpy.uintc, ndim = 3, flags='CONTIGUOUS')
int_2d_array          = numpy.ctypeslib.ndpointer(dtype = numpy.intc,  ndim = 2, flags='CONTIGUOUS')


_liblookuptable.lookuptable.argtypes = [int_2d_array,
                                        ctypes.c_uint,
                                        ctypes.c_uint,
                                        unsigned_int_3d_array,
                                        unsigned_int_3d_array,
                                        ctypes.c_uint]

def build_lookuptable(**kwargs):
    data, size = kwargs['data'], kwargs['size']
    assert data and size
    lut = lookuptable()
    lut.build(data, size)
    return lut

class lookuptable(object):
    def __init__(self, table = None):
        self.table = table

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


    def build(self, data = None, size = None):
        assert data and size
        self.size = size
        count = numpy.zeros((size, size, size), dtype = 'uintc')
        lookuptable = numpy.zeros((size, size, size), dtype = 'uintc')

        values = (numpy.round((numpy.copy(data) * self.size - 0.5))).astype('uintc') # redefine values to be used properly as indices
        values[values >= self.size] = self.size - 1 # make sure none of the values exceed 1, if they do simply set them to the max.

        shape = values.shape.astype('uintc')
        _liblookuptable.lookuptable(values, shape[0],  shape[1], lookuptable, count, numpy.asarray([size,], dtype = 'uintc')[0])
        return lookuptable, count






