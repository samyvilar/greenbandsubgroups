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

_liblookuptable.set_min_max.argtypes = [int_2d_array,
                                        ctypes.c_uint,
                                        ctypes.c_uint,
                                        float_3d_array,
                                        ctypes.c_uint,
                                        ctypes.c_uint]

_liblookuptable.update_min_max_lut.argtypes = [float_3d_array,
                                               float_3d_array,
                                               ctypes.c_uint,
                                               ctypes.c_uint]

def update_min(prev_mins = None, new_mins = None, lut_size = None, method = 'C'):
    assert prev_mins != None and new_mins != None and lut_size != None
    if method == 'C':
        _liblookuptable.update_min_max_lut(prev_mins, new_mins, lut_size, 0)
    elif method == 'numpy':
        prev_mins.resize(lut_size, lut_size, lut_size, 1)
        new_mins.resize(lut_size, lut_size, lut_size, 1)
        prev_mins = numpy.concatenate((prev_mins, new_mins), axis = 3).min(axis = 3)
    else:
        raise ValueError("Expected either method 'C' or 'numpy'")

    return prev_mins

def update_max(prev_max = None, new_max = None, lut_size = None):
    assert prev_max != None and new_max != None and lut_size != None
    _liblookuptable.update_min_max_lut(prev_max, new_max, lut_size, 1)

def build_lookuptable(kwvalues):
    lut = lookuptable()
    lut.build(**kwvalues)
    return lut


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
            self.max_value = 1.0
        elif any(['_lookuptable_' in file for file in os.listdir(lookuptable_path)]):
            file  = [file for file in os.listdir(lookuptable_path) if '_lookuptable_' in file and '_lookuptable_flatten_' not in file]
            if not file:
                raise ValueError("Couldn't find the lookuptable withing %s" % lookuptable_path)
            self.table = numpy.fromfile(lookuptable_path + '/' + file[0])
            self.size = int(basename(file[0]).split('_')[0])
            self.max_value = float(basename(file[0]).split('_')[-2])
        else:
            raise Exception("%s is not a proper granule path or directory!" % lookuptable_path)
        self.table = self.table.reshape((self.size, self.size, self.size))


    def load_flatten_table(self, lookuptable_flatten_path):
        if '.numpy' in lookuptable_flatten_path:
            self.flatten_table = numpy.fromfile(lookuptable_flatten_path)
            self.size = int(basename(lookuptable_flatten_path).split('_')[0])
            self.max_value = 1
        elif any(['_lookuptable_flatten_' in file for file in os.listdir(lookuptable_flatten_path)]):
            flatten_file = [file for file in os.listdir(lookuptable_flatten_path) if '_lookuptable_flatten_' in file]
            self.flatten_table = numpy.fromfile(lookuptable_flatten_path + '/' + flatten_file[0])
            self.size = int(basename(flatten_file[0]).split('_')[0])
            self.max_value = float(basename(flatten_file[0]).split('_')[-2])
        else:
            raise Exception('%s is not a proper granule path or directory!' % lookuptable_flatten_path)
        self.flatten_table = self.flatten_table.reshape((self.flatten_table.shape[0]/4, 4))

    @property
    def max_value(self):
        return self._max_value
    @max_value.setter
    def max_value(self, value):
        self._max_value = value

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
        indices = self.data_to_indices(granule, self.max_value)

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
        assert max_value > 0
        values = (numpy.round( ((data/max_value) * self.size - 0.5) )).astype('intc') # redefine values to be used properly as indices
        values[values >= self.size] = self.size - 1 # make sure none of the values exceed max, if they do simply set them to the max.
        values[values < 0] = 0

        return values

    def indices_to_data(self, indices):
        return ((indices + 0.5)/self.size)*self.max_value

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





    def build(self, data = None, size = None, max_value = None, function = None):
        assert data != None and size != None
        self.size = size
        values = self.data_to_indices(data, max_value = max_value)
        shape = numpy.asarray(data.shape, dtype = 'uintc')
        if function == None:
            assert max_value != None
            self.max_value = max_value
            counts = numpy.zeros((size, size, size), dtype = 'float32')
            sums = numpy.zeros((size, size, size), dtype = 'float32')

            _liblookuptable.lookuptable(values,
                                        shape[0],
                                        shape[1],
                                        sums,
                                        counts,
                                        numpy.asarray([size,], dtype = 'uintc')[0])

            self.counts = counts
            self.sums = sums
        elif function == 'min':
            mins = numpy.zeros((size, size, size), dtype = 'float32')
            mins[:] = size + 100
            _liblookuptable.set_min_max(values,
                                        shape[0],
                                        shape[1],
                                        mins,
                                        numpy.asarray([size,], dtype = 'uintc')[0],
                                        0)
            self.min = mins
        elif function == 'max':
            max = numpy.zeros((size, size, size), dtype = 'float32')
            max[:] = -1
            _liblookuptable.set_min_max(values,
                                        shape[0],
                                        shape[1],
                                        max,
                                        numpy.asarray([size,], dtype = 'uintc')[0],
                                        1)
            self.max = max
        else:
            raise Exception("function can only be min or max! got %s" % function)


        gc.collect()



def get_best_bands(data_set = None, bands = [0,1,2,3,4,5]):
    pass






