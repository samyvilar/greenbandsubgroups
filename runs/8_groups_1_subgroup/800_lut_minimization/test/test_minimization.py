__author__ = 'samyvilar'



import sys
import pickle
import numpy
import socket
sys.path.extend('../../../..')

from MeanCalculator import get_predicted_from_means
from lookuptable.lookuptable import  lookuptable
from Utils import save_images
from GranuleLoader import GranuleLoader


if __name__ == '__main__':
    all_means = pickle.load(open('../all_means.obj', 'rb'))
    best_mean_index = numpy.asarray(pickle.load(open('../sum_of_errors.obj', 'rb'))).argmin()
    means = all_means[best_mean_index]


    lut = lookuptable()
    lut.load_flatten_table('../../../lookuptable/reflectance/800/800_lookuptable_flatten.numpy')
    lut_data_flatten = lut.indices_to_data(lut.flatten_table)

    granule_loader = GranuleLoader()
    granule_loader.bands = [1,2,3,4]
    granule_loader.param = 'reflectance'
    granule_loader.disable_caching()
    granule_loader.disable_multithreading()


    if socket.gethostname() == 'fermi.localdomain':
        granule_path = '/DATA_5/SNOW_CLOUD_MODIS/data/MOD021KM.A2002179.1640.005.2010085164818.hdf'
    else:
        granule_path = '/home1/FermiData/DATA_5/SNOW_CLOUD_MODIS/data/MOD021KM.A2002179.1640.005.2010085164818.hdf'

    granule_loader.load_granules(granules = [granule_path,])
    test_data = granule_loader.granules[0].data
    predicted = get_predicted_from_means(data = lut_data_flatten,
                                         means = means,
                                         original = test_data,
                                         training_band = [3],
                                         predictive_band = [0,1,2],
                                         enable_multithreading = False)


    save_images(original = original,
        predicted = predicted,
        granule_path = granule_path,
        original_shape = granule_loader.granules[0].original_shape)

    error = get_root_mean_square(original = original[:, 3], predicted = predicted[:, 3])
    print "RMSE: %f%%" % error
    print "Minimized mean index %i" % best_mean_index




