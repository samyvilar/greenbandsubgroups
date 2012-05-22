__author__ = 'Samy Vilar'

import sys
sys.path.extend('../../../../../..')

from lookuptable.lookuptable import lookuptable
from GranuleLoader import GranuleLoader
from Utils import save_images, get_root_mean_square, get_sum_of_errors_squared, get_granule_path

if __name__ == '__main__':
    lut = lookuptable()
    lut.load_table('../800_lookuptable.numpy')

    granule_loader = GranuleLoader()
    granule_loader.bands = [1,2,3,4]
    granule_loader.param = 'reflectance'
    granule_loader.disable_caching()
    granule_loader.enable_multithreading()

    granule_path = get_granule_path() + 'MOD021KM.A2002179.1640.005.2010085164818.hdf'
    granule_loader.load_granules(granules = [granule_path,])
    original = granule_loader.granules[0].data
    predicted = lut.predict(original)

    error = get_root_mean_square(original = original[:, 3], predicted = predicted[:, 3])
    print "RMSE: %f%%" % error
    print "Sum of Squared Errors: %f" % get_sum_of_errors_squared(original = original[:, 3], predicted = predicted[:, 3])
    save_images(original = original, predicted = predicted, granule_path = granule_path, original_shape = granule_loader.granules[0].original_shape)


# red is 1, green = 4, blue = 3, NIR = 2
# 1, 4, 3
# 0, 3, 2


