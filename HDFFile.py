__author__ = 'Samy Vilar'

import glasslab_cluster.io
import numpy


def read_file(file = None, bands = None, param = None, crop_size = None, crop_orig = None):
    data = []
    valid_range  = numpy.zeros((len(bands), 2))
    for index, band in enumerate(bands):
        mod = glasslab_cluster.io.modis_level1b_read(file, band, param = param, clean = True)
        if numpy.any(mod['mimage'].mask):
            print "flags exist in band %d of granule %s" % (band, os.path.basename(file))
            raise Exception('Bad Granule')
        img = numpy.asarray(mod['mimage'].data, dtype = 'float').copy()
        valid_range[index,:] = mod['validrange']
        n       = numpy.sum((valid_range[index, 0] > img) | (img > valid_range[index, 1]))
        if n > 0:
            print "Valid Range:", mod['validrange'], " %d values out of valid range in band %d" % (n, band)
            raise Exception('Bad Granule')

        if param == 'reflectance': #(path, band, mimage, param='radiance', start=None)
            os.system("cp " + file + " " + "/tmp/")
            temp_file = "/tmp/" + os.path.basename(file)
            glasslab_cluster.io.modis_level1b_write(temp_file, band, img, param = param)
            img = glasslab_cluster.io.modis_crefl(temp_file, bands = [band,])[0]
            os.system("rm " + "/tmp/" + os.path.basename(file))

        data.append(img)
    data = numpy.dstack(data)
    if crop_orig and crop_size:
        crop = data[crop_orig[0]:crop_orig[0] + crop_size[0], crop_orig[1]:crop_orig[1] + crop_size[1]]
    else:
        crop = data
    return crop.reshape(crop.shape[0] * crop.shape[1], len(bands)).astype(numpy.dtype('f8')), valid_range

class HDFFile(object):
    def __int__(self, file):
        self.set_file(file)

    def set_file(self, file):
        self._file = file
    def get_file(self):
        return self._file

    def set_param(self, param):
        self._param = param
    def get_param(self):
        return self._param

    def set_crop_size(self, crop_size):
        self._crop_size = crop_size
    def get_crop_size(self):
        return self._crop_size

    def set_crop_orig(self, crop_orig):
        self._crop_orig = crop_orig
    def get_crop_orig(self):
        return self._crop_orig

    def set_bands(self, bands):
        self._bands = bands
    def get_bands(self, bands):
        return self._bands

    def set_data(self, data):
        self._data = data
    def get_data(self):
        return self._data

    def set_valid_range(self, valid_range):
        self._valid_range = valid_range
    def get_valid_range(self):
        return self._valid_range

    def load(self):
        try:
            data, valid_range = read_file(file = self.get_file(), bands = self.get_bands(), param = self.get_param(), crop_size = self.get_crop_size(), crop_orig = self.get_crop_orig())
            self.set_data(data)
            self.set_valid_range(valid_range)
            return self
        except Exception as ex:
            return None

