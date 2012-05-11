__author__ = 'samyvilar'

import os
import inspect
import ctypes
import numpy

_liblabelling = numpy.ctypeslib.load_library('liblabelling', os.path.dirname(inspect.getfile(inspect.currentframe())))

double_3d_array = numpy.ctypeslib.ndpointer(dtype = numpy.float64, ndim = 3, flags = 'CONTIGUOUS')
double_2d_array = numpy.ctypeslib.ndpointer(dtype = numpy.float64, ndim = 2, flags = 'CONTIGUOUS')
double_1d_array = numpy.ctypeslib.ndpointer(dtype = numpy.float64, ndim = 1, flags = 'CONTIGUOUS')
'''
void set_labels(double *data, unsigned int data_number_of_rows, unsigned int data_number_of_columns,
               double *means, unsigned int number_of_sub_groups, unsigned int means_number_of_rows, unsigned int means_number_or_columns,
               unsigned int *labels)
'''
_liblabelling.set_labels.argtypes = [double_2d_array,
                                     ctypes.c_uint,
                                     ctypes.c_uint,
                                     double_2d_array,
                                     ctypes.c_uint,
                                     ctypes.c_uint,
                                     ctypes.c_uint,
                                     double_1d_array
                                     ]

def get_labels(**kwargs):
    data = kwargs['data']
    means = kwargs['means']
    if len(means.shape) == 2:
        labels = numpy.zeros(data.shape[0], dtype = 'float64')
        _liblabelling.set_labels(data,
                                 data.shape[0],
                                 data.shape[1],
                                 means,
                                 1,
                                 means.shape[0],
                                 means.shape[1],
                                 labels)
        return labels 
