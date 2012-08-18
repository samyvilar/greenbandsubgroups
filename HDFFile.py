__author__ = 'Samy Vilar'

from glasslab_cluster.io import modis
import numpy
import os
import shutil


def read_file(file = None, bands = None, param = None, crop_size = None, crop_orig = None):
    assert file and bands and param and (param == 'reflectance' or param == 'radiance')
    data = []
    valid_range  = numpy.zeros((len(bands), 2))

    if param == 'reflectance':
        temp_file = '/home1/student/svilar/Development/tmp/temp_' + os.path.basename(file)
        shutil.copyfile(file, temp_file)
    elif param == 'radiance':
        temp_file = file
    else:
        raise Exception("Param wasn't set to 'reflectance' or 'radiance' got '%s'" % str(param))

    granule = modis.Level1B(temp_file, mode = 'w')
    for index, band in enumerate(bands):
        if param == 'reflectance':
            b = granule.reflectance(band)
        elif param == 'radiance':
            b = granule.radiance(band)
        else:
            raise Exception("Param wasn't set to 'reflectance' or 'radiance' got '%s'" % str(param))

        b.write(b.fill_invalid(b.read()))
        b.close()

        if param == 'reflectance':
            b = granule.reflectance(band)
        elif param == 'radiance':
            b = granule.radiance(band)

        data.append(b.read())
        b.close()

    granule.close()
    if temp_file.startswith('/home1/student/svilar/Development/tmp'):
        os.remove(temp_file)
    if param == 'reflectance':
        data = modis.crefl(temp_file, bands = bands)

    data = numpy.dstack(data)

    if crop_orig and crop_size:
        data = data[crop_orig[0]:crop_orig[0] + crop_size[0], crop_orig[1]:crop_orig[1] + crop_size[1]]

    original_shape = data.shape
    return data.reshape(data.shape[0] * data.shape[1], len(bands)).astype('float64'), valid_range, original_shape

class HDFFile(object):
    def __init__(self, file):
        self._bands          = []
        self._param          = None
        self._crop_size      = None
        self._crop_orig      = None
        self._file_name      = None
        self._file_dir       = None
        self._data           = None
        self._valid_range    = None
        self._original_shape = None
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
    def original_shape(self):
        return self._original_shape
    @original_shape.setter
    def original_shape(self, shape):
        self._original_shape = shape

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
        self.data, self.valid_range, self.original_shape = read_file(file = self.file,
                                                bands = self.bands,
                                                param = self.param,
                                                crop_size = self.crop_size,
                                                crop_orig = self.crop_orig)
        return self


