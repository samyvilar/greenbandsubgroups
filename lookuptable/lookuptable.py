__author__ = 'Samy Vilar'
__date__ = 'Mar 15, 2012'

import numpy
import numpy.ctypeslib
import ctypes

import os.path
import inspect
import gc
gc.enable()

from os.path import basename
from Utils import multithreading_pool_map
import numpy

_liblookuptable = numpy.ctypeslib.load_library('liblut', os.path.dirname(inspect.getfile(inspect.currentframe())))

float_3d_array = numpy.ctypeslib.ndpointer(dtype = numpy.float32, ndim = 3, flags = 'CONTIGUOUS')
int_2d_array   = numpy.ctypeslib.ndpointer(dtype = numpy.intc,    ndim = 2, flags = 'CONTIGUOUS')

double_3d_array = numpy.ctypeslib.ndpointer(dtype = numpy.float64, ndim = 3, flags = 'CONTIGUOUS')
double_2d_array = numpy.ctypeslib.ndpointer(dtype = numpy.float64, ndim = 2, flags = 'CONTIGUOUS')
double_1d_array = numpy.ctypeslib.ndpointer(dtype = numpy.float64, ndim = 1, flags = 'CONTIGUOUS')

_liblookuptable.lookuptable.argtypes = [int_2d_array,
                                        ctypes.c_uint,
                                        ctypes.c_uint,
                                        float_3d_array,
                                        float_3d_array,
                                        ctypes.c_uint]

_liblookuptable.predict_double.argtypes = [int_2d_array,
                                           ctypes.c_uint,
                                           ctypes.c_uint,
                                           double_3d_array,
                                           ctypes.c_uint,
                                           double_1d_array]

_liblookuptable.flatten_lookuptable.argtypes = [double_3d_array,
                                                ctypes.c_uint,
                                                double_2d_array,
                                                ctypes.c_uint]

def build_lookuptable(kwvalues):
    data, size, max_value = kwvalues['data'], kwvalues['size'], kwvalues['max_value']
    assert data != None and size != None
    lut = lookuptable()
    lut.build(data, size, max_value = max_value)
    return lut

def find_max(granule_loader = None):
    assert granule_loader
    max_value = numpy.max(multithreading_pool_map(
        **{
            'values':granule_loader.next(),
            'function':numpy.max,
            'multithreaded':True
        }))

    for granule_chunks in granule_loader:
        temp_max = numpy.max(multithreading_pool_map(
          **{

            }))

class lookuptable(object):
    def __init__(self):
        self._sums = None
        self._counts = None
        self._max_value = 1

    def enable_multithreading(self):
        self._multithreading = True
    def disable_multithreading(self):
        self._multithreading = False

    def enable_caching(self):
        self._caching = True
    def disable_caching(self):
        self._caching = False


    def load_table(self, lookuptable_path):
        if '.numpy' in lookuptable_path:
            self.table = numpy.fromfile(lookuptable_path)
            self.size = int(basename(lookuptable_path).split('_')[0])
            self.table = self.table.reshape((self.size, self.size, self.size))
            self.max_value = 1
        elif any('_lookuptable_' in file for file in os.listdir(lookuptable_path)):
            file  = [file for file in os.listdir(lookuptable_path) if '_lookuptable_' in file and '_lookuptable_flatten_' not in file]
            if not file:
                raise ValueError("Couldn't find the lookuptable withing %s" % lookuptable_path)
            self.table = numpy.fromfile(lookuptable_path + '/' + file[0])
            self.size = int(basename(file[0]).split('_')[0])
            self.max_value = float(basename(file[0]).split('_')[-1])
        self.table = self.table.reshape((self.size, self.size, self.size))


    def load_flatten_table(self, lookuptable_flatten_path):
        self.flatten_table = numpy.fromfile(lookuptable_flatten_path)
        self.flatten_table = self.flatten_table.reshape((self.flatten_table.shape[0]/4, 4))
        self.size = int(basename(lookuptable_flatten_path).split('_')[0])

    @property
    def max_value(self):
        return self._max_value
    @max_value.setter
    def max_value(self, value):
        self._max_value = max_value

    @property
    def table(self):
        return self._table
    @table.setter
    def table(self, value):
        self._table = value

    @property
    def flatten_table(self):
        return self._flatten_table
    @flatten_table.setter
    def flatten_table(self, flatten_table):
        self._flatten_table = flatten_table

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

    def predict(self, granule):
        prediction_green = numpy.zeros(granule.shape[0], dtype = 'float64')
        indices = self.data_to_indices(granule)

        shape = numpy.asarray(indices.shape, dtype = 'uintc')
        _liblookuptable.predict_double(indices,
                                       shape[0],
                                       shape[1],
                                       self.table,
                                       numpy.asarray([self.size,], dtype = 'uintc')[0],
                                       prediction_green)

        prediction = numpy.zeros(granule.shape)
        prediction[:, 0:3] = granule[:, 0:3]
        prediction[:, 3] = self.indices_to_data(prediction_green)
        return prediction



    def data_to_indices(self, data, max_value):
        values = (numpy.round((data * (self.size/max_value) - 0.5))).astype('intc') # redefine values to be used properly as indices
        values[values >= self.size] = self.size - 1 # make sure none of the values exceed max, if they do simply set them to the max.
        values[values < 0] = 0

        return values

    def indices_to_data(self, indices):
        return ((indices + 0.5)*self._max_value)/self.size

    def flatten_2d_non_zero(self):
        assert self.table != None
        non_zero_count = numpy.sum(self.table != 0)
        lookuptable_flatten = numpy.zeros((non_zero_count, 4))

        _liblookuptable.flatten_lookuptable(self.table,
                                            numpy.asarray([self.size,], dtype = 'uintc')[0],
                                            lookuptable_flatten,
                                            ctypes.c_uint(non_zero_count))
        return lookuptable_flatten

    def load_or_calculate_flatten_table(self, table_path):
        assert self.table != None
        if (os.path.isfile(table_path)):
            flatten_table = numpy.fromfile(table_path)
            self._flatten_table = flatten_table.reshape((flatten_table.shape[0]/4, 4))
        else:
            self._flatten_table = self.flatten_2d_non_zero()





    def build(self, data = None, size = None, max_value = 1):
        assert data != None and size != None
        self.size = size
        counts = numpy.zeros((size, size, size), dtype = 'float32')
        sums = numpy.zeros((size, size, size), dtype = 'float32')

        values = self.data_to_indices(data, max_value = max_value)

        shape = numpy.asarray(data.shape, dtype = 'uintc')
        _liblookuptable.lookuptable(values,
                                    shape[0],
                                    shape[1],
                                    sums,
                                    counts,
                                    numpy.asarray([size,], dtype = 'uintc')[0])

        self.counts = counts
        self.sums = sums
        #self.table = numpy.zeros((size, size, size), dtype = 'float32')
        #non_zero_locations = counts != 0
        #self.table[non_zero_locations] = sums[non_zero_locations]/counts[non_zero_locations]

        #del non_zero_locations
        gc.collect()







