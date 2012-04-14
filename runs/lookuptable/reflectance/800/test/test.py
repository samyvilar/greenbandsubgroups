__author__ = 'Samy Vilar'

import sys
sys.path.extend('../../../../../..')

import numpy

from lookuptable.lookuptable import lookuptable
from GranuleLoader import GranuleLoader
from Utils import save_images, get_root_mean_square

if __name__ == '__main__':
    lut = lookuptable()
    lut.load_table('../800_lookuptable.numpy')

    granule_loader = GranuleLoader()
    granule_loader.bands = [1,2,3,4]
    granule_loader.param = 'reflectance'
    granule_loader.disable_caching()
    granule_loader.enable_multithreading()

    granule_path = '/DATA_5/SNOW_CLOUD_MODIS/data/MOD021KM.A2002179.1640.005.2010085164818.hdf'
    granule_loader.load_granules(granules = [granule_path,])
    original = granule_loader.granules[0].data
    predicted = lut.predict(original)

    error = get_root_mean_square(original = original[:, 3], predicted = predicted[:, 3])
    print "RMSE: %f%%" % error

    save_images(original = original, predicted = predicted, granule_path = granule_path, original_shape = granule_loader.granules[0].original_shape)


# red is 1, green = 4, blue = 3, NIR = 2
# 1, 4, 3
# 0, 3, 2


