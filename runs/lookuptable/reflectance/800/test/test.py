__author__ = 'Samy Vilar'

import sys
sys.path.extend('../../../../../..')

import numpy

from lookuptable.lookuptable import lookuptable
from GranuleLoader import GranuleLoader
from Utils import image_show
from os.path import basename

if __name__ == '__main__':
    lut = lookuptable()
    lut.load_table('../800_lookuptable.numpy')

    granule_loader = GranuleLoader()
    granule_loader.bands = [1,2,3,4]
    granule_loader.param = 'reflectance'
    granule_loader.disable_caching()
    granule_loader.enable_multithreading()

    granule_path = '/home1/FermiData/DATA_5/SNOW_CLOUD_MODIS/data/MOD021KM.A2002179.1640.005.2010085164818.hdf'
    granule_loader.load_granules(granules = [granule_path,])
    original = granule_loader.granules[0].data
    predicted = lut.predict(original)

    std = numpy.sqrt(numpy.sum((predicted[:, 3] - original[:, 3])**2)/original[:, 3].shape[0])
    print std/numpy.mean(original[:,3]) * 100

    original_shape = granule_loader.granules[0].original_shape[0:2]
    original_green = original[:, 3].reshape(original_shape)
    predicted_green = predicted[:, 3].reshape(original_shape)

    image_show(source = original_green,
        vmin = 0, vmax = 1, min = 0, max = 1,
        interpolation = 'nearest',
        color_bar = True, file_name = 'original_green.png',
        title = 'True Green Values granule %s' % basename(granule_path))

    image_show(source = predicted_green,
        vmin = 0, vmax = 1, min = 0, max = 1,
        interpolation = 'nearest',
        color_bar = True, file_name = 'predicted_green.png',
        title = 'Predicted Green Values granule %s' % basename(granule_path))

    image_show(source = numpy.fabs(original_green - predicted_green),
        vmin = 0, vmax = 1, min = 0, max = 1,
        interpolation = 'nearest',
        color_bar = True, file_name = 'error.png',
        title = 'Absolute Error of Green Values Predicted vs True')

    image_show(source = numpy.fabs(original_green - predicted_green)/original_green,
        vmin = 0, vmax = 1, min = 0, max = 1,
        interpolation = 'nearest',
        color_bar = True, file_Name = 'relatibe_error.png',
        title = 'Relative Error of Green Values Predicted vs True')


    image_show(
       source = granule_loader.granules[0].data,
       reshape = granule_loader.granules[0].original_shape,
       vmin = 0, vmax = 1, min = 0, max = 1,
       interpolation = 'nearest',
       color_bar = True, file_name = 'original_rgb_type_casted.png',
       title = 'True RGB granule %s' % basename(granule_path),
       red_index = 0, green_index = 3, blue_index = 2)

    image_show(source = predicted,
        reshape = granule_loader.granules[0].original_shape,
        vmin = 0, vmax = 1,
        interpolation = 'nearest',
        color_bar = True, file_name = 'predicted_rgb_type_casted',
        title = 'Predicted RGB granule %s' % basename(granule_path),
        red_index = 0, green_index = 3, blue_index = 2)

    image_show(source = granule_loader.granules[0].data,
        reshape = granule_loader.granules[0].original_shape,
        vmin = 0, vmax = 1, min = 0, max = 1,
        interpolation = 'nearest',
        crop_origin = (500, 0), crop_size = (750, 600),
        color_bar = True, file_name = 'original_rgb_type_casted_500-1250_0-600.png',
        title = 'True RGB type casted and cropped granule %s' % basename(granule_path),
        red_index = 0, green_index = 3, blue_index = 2)


    image_show(source = predicted,
        reshape = granule_loader.granules[0].original_shape,
        vmin = 0, vmax = 1,
        interpolation = 'nearest',
        crop_origin = (500, 0), crop_size = (750, 600),
        color_bar = True, file_name = 'predicted_rgb_type_casted_500-1250_0-600.png',
        title = 'Predicted RGB type casted and cropped granule %s' % basename(granule_path),
        red_index = 0, green_index = 3, blue_index = 2)


# red is 1, green = 4, blue = 3, NIR = 2
# 1, 4, 3
# 0, 3, 2


