__author__ = 'Samy Vilar'

from glasslab_cluster.io import modis
import shutil
import os
import numpy
from matplotlib import pyplot as plt


def test_clean_crefl(granule_path, bands = [1,2,2,4]):
    temp_file = os.path.join(os.getcwd(), 'temp_' + os.path.basename(granule_path))
    shutil.copy(granule_path, temp_file)

    g_read = modis.Level1B(granule_path, mode = 'r')
    g_write = modis.Level1B(temp_file, mode = 'w')

    original = []
    for band in bands:
        b_read = g_read.reflectance(band)
        b_write = g_write.reflectance(band)

        img = b_read.read()
        b_write.write(
            b_read.destripe(
                b_read.fill_invalid(
                    img, winsize = 75))) # b.read() only supports clean, but really should have options to set winsize and ...
        b_write.close()
        b_read.close()
        original.append(img)

    g_read.close()
    g_write.close()

    plt.imshow()
    crefl_data = modis.crefl(temp_file, bands = bands)

    data = numpy.dstack(data)
    crefl_data = numpy.dstack(crefl_data )

    for index in range(len(bands)):
        plt.figure()
        plt.imshow(original[:, index], vmin = 0, min = 0, vmax = 1, max = 1, interpolation = 'nearest')
        plt.colorbar()
        plt.savefig('original_band_%i.png' % (index + 1))
        plt.figure()
        plt.imshow(crefl_data[:, index], vmin = 0, min = 0, vmax = 1, max = 1, interpolation = 'nearest')
        plt.colorbar()
        plt.savefig('corrected_band_%i.png' % (index + 1))



if __name__ == "__main__":
    test_clean_crefl("/tmp/MOD021KM.A2002179.1640.005.2010085164818.hdf")













