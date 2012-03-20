__author__ = 'Samy Vilar'
__date__ = 'Mar 15, 2012'

import numpy
import numpy.ctypeslib
import ctypes

import os.path
import inspect


_liblookuptable = numpy.ctypeslib.load_library('liblut', os.path.dirname(inspect.getfile(inspect.currentframe())))

unsigned_int_3d_array    = numpy.ctypeslib.ndpointer(dtype = numpy.uintc,    ndim = 3, flags='CONTIGUOUS')
unsigned_int_2d_array    = numpy.ctypeslib.ndpointer(dtype = numpy.uintc,    ndim = 2, flags='CONTIGUOUS')
double_1d_array = numpy.ctypeslib.ndpointer(dtype = numpy.double,  ndim = 1, flags='CONTIGUOUS')


_liblookuptable.lookuptable.argtypes = [unsigned_int_2d_array,
                                        ctypes.c_uint,
                                        ctypes.c_uint,
                                        unsigned_int_3d_array,
                                        unsigned_int_3d_array,
                                        ctypes.c_uint]
#_liblookuptable.lookuptable.restype = ctypes.c_void_p


#void predict(int *data, int numrows, int numcols, double *validrange, int *lookuptable,                     int lutsize, double *results)
_liblookuptable.predict.argtypes = [int_2d_array, ctypes.c_int, ctypes.c_int, double_1d_array, int_3d_array, ctypes.c_int, double_1d_array, ctypes.c_int]
#_liblookuptable.predict.restype =



class LookUpTable(object):
    def __ini(self, table = None):
        self.table = table

    @property
    def table(self):
        return self._table
    @table.set
    def table(self, value):
        self._table = value

    @property
    def size(self):
        return self._size
    @size.set
    def size(self, value):
        self._size = value


    def build(self, data = None, size = None, file = None):
        assert data and size
        self.size = size
        values = (numpy.round((numpy.copy(data) * self.size - 0.5))).astype('uintc') # redefine values to be used properly as indices
        values[values > self.size] = self.size # make sure none of the values exceed 1, if they do simply set them to the max.

        _liblookuptable.lookuptable(values, values.shape[0], values.shape[1], lut, count, lutsize);





