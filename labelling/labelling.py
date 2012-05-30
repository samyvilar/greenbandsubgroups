__author__ = 'samyvilar'

import os
import inspect
import ctypes
import numpy

_liblabelling = numpy.ctypeslib.load_library('liblabelling', os.path.dirname(inspect.getfile(inspect.currentframe())))

double_3d_array = numpy.ctypeslib.ndpointer(dtype = numpy.float64, ndim = 3, flags = 'CONTIGUOUS')
double_2d_array = numpy.ctypeslib.ndpointer(dtype = numpy.float64, ndim = 2, flags = 'CONTIGUOUS')
double_1d_array = numpy.ctypeslib.ndpointer(dtype = numpy.float64, ndim = 1, flags = 'CONTIGUOUS')
uint_1d_array   = numpy.ctypeslib.ndpointer(dtype = numpy.uint32,    ndim = 1, flags = 'CONTIGUOUS')
uint_2d_array   = numpy.ctypeslib.ndpointer(dtype = numpy.uint32,    ndim = 2, flags = 'CONTIGUOUS')

'''
void set_labels(double *data, unsigned int data_number_of_rows, unsigned int data_number_of_columns,
               double *means, unsigned int number_of_sub_groups, unsigned int means_number_of_rows, unsigned int means_number_or_columns,
               unsigned int *labels)
'''

def get_labels(**kwargs):
    data = kwargs['data']
    means = kwargs['means']
    if len(means.shape) == 2:
        _liblabelling.set_labels.argtypes = [double_2d_array,
                                             ctypes.c_uint,
                                             ctypes.c_uint,
                                             double_2d_array,
                                             ctypes.c_uint,
                                             ctypes.c_uint,
                                             ctypes.c_uint,
                                             uint_1d_array
        ]

        labels = numpy.zeros(data.shape[0], dtype = 'uint32')
        print 'data.dtype ' + str(data.dtype)
        print 'data.shape[0] ' + str(data.shape[0])
        print 'data.shape[1] ' + str(data.shape[1])
        _liblabelling.set_labels(data,
                                 data.shape[0],
                                 data.shape[1],
                                 means,
                                 1,
                                 means.shape[0],
                                 means.shape[1],
                                 labels)
        return labels
    elif len(means.shape) == 3:
        _liblabelling.set_labels.argtypes = [double_2d_array,
                                             ctypes.c_uint,
                                             ctypes.c_uint,
                                             double_3d_array,
                                             ctypes.c_uint,
                                             ctypes.c_uint,
                                             ctypes.c_uint,
                                             uint_2d_array
        ]

        labels = numpy.zeros((data.shape[0], 2), dtype = 'uint32')
        _liblabelling.set_labels(data,
            data.shape[0],
            data.shape[1],
            means,
            means.shape[0],
            means.shape[1],
            means.shape[2],
            labels)
        return labels
    else:
        raise Exception("Only Supporting groups and subgroups!")
