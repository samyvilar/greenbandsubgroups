__author__ = 'Samy Vilar'

import sys
sys.path.extend('../../../../../..')

from lookuptable.lookuptable import lookuptable
from HDFFile import read_file
from Utils import save_images, get_root_mean_square, get_sum_of_errors_squared, get_granule_path

if __name__ == '__main__':
    lut = lookuptable()
    lut.load_table('../800_lookuptable.numpy')

    granule_path = '/home1/FermiData/DATA_3/SNOW_CLOUD_MODIS/data/MOD021KM.A2002179.1640.005.2010085164818.hdf'
    original_clean, valid_range, origina_shape = read_file(file = granule_path,
                         bands = [1,2,3,4],
                         param = 'reflectance',
                         winsize = 75,
                         maxinvalid = .35,
                         clean = True)

    predicted = lut.predict(original)

    error = get_root_mean_square(original = original[:, 3], predicted = predicted[:, 3])
    print "RMSE: %f%%" % error
    print "Sum of Squared Errors: %f" % get_sum_of_errors_squared(original = original[:, 3], predicted = predicted[:, 3])
    save_images(original = original, predicted = predicted, granule_path = granule_path, original_shape = origina_shape )


# red is 1, green = 4, blue = 3, NIR = 2
# 1, 4, 3
# 0, 3, 2


