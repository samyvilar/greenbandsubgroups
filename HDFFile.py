__author__ = 'Samy Vilar'

import glasslab_cluster.io
import numpy
import os


def read_file(file = None, bands = None, param = None, crop_size = None, crop_orig = None):
    assert file and bands and param
    data = []
    valid_range  = numpy.zeros((len(bands), 2))

    if param == 'radiance':
        for index, band in enumerate(bands):
            mod = glasslab_cluster.io.modis_level1b_read(file, band, param = param, clean = True)

            if numpy.any(mod['mimage'].mask):
                print "flags exist in band %d of granule %s" % (band, os.path.basename(file))
                raise Exception('Bad Granule')
            img = numpy.asarray(mod['mimage'].data, dtype = 'float')
            valid_range[index, :] = mod['validrange']
            n  = numpy.sum((valid_range[index, 0] > img) | (img > valid_range[index, 1]))
            if n > 0:
                print "Valid Range:", mod['validrange'], " %d values out of valid range in band %d" % (n, band)
                raise Exception('Bad Granule')

            data.append(img)
    elif param == 'reflectance':
        data = glasslab_cluster.io.modis_crefl(file, bands = bands).astype('float32')

    data = numpy.dstack(data)
    if crop_orig and crop_size:
        data = data[crop_orig[0]:crop_orig[0] + crop_size[0], crop_orig[1]:crop_orig[1] + crop_size[1]]

    return data.reshape(data.shape[0] * data.shape[1], len(bands)), valid_range

class HDFFile(object):
    def __init__(self, file):
        self._bands         = []
        self._param         = None
        self._crop_size     = None
        self._crop_orig     = None
        self._file_name     = None
        self._file_dir      = None
        self._data          = None
        self._valid_range   = None
        self.file = file

    @property
    def param(self):
        return self._param
    @param.setter
    def param(self, value):
        self._param = value
    @property
    def bands(self):
        return self._bands
    @bands.setter
    def bands(self, values):
        self._bands = values
    @property
    def crop_size(self):
        return self._crop_size
    @crop_size.setter
    def crop_size(self, values):
        self._crop_size = values
    @property
    def crop_orig(self):
        return self._crop_orig
    @crop_orig.setter
    def crop_orig(self, values):
        self._crop_orig = values

    @property
    def file(self):
        return self._file
    @file.setter
    def file(self, value):
        self._file      = value
        self.file_name  = os.path.basename(self.file)
        self.file_dir   = os.path.dirname(self.file)

    @property
    def file_name(self):
        return self._file_name
    @file_name.setter
    def file_name(self, values):
        self._file_name = values
    @property
    def file_dir(self):
        return self._file_dir
    @file_dir.setter
    def file_dir(self, values):
        self._file_dir = values
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, values):
        self._data = values

    @property
    def valid_range(self):
        return self._valid_range
    @valid_range.setter
    def valid_range(self, values):
        self._valid_range = values

    def _verify_properties(self):
        assert self.param and len(self.bands) > 0 and self.file

    def load(self):
        self._verify_properties()
        self.data, self.valid_range = read_file(file = self.file,
                                                bands = self.bands,
                                                param = self.param,
                                                crop_size = self.crop_size,
                                                crop_orig = self.crop_orig)
        return self


