__author__ = 'Samy Vilar'
__date__ = 'Mar 22, 2012'

import sys
sys.path.extend('../../..')

from lookuptable.lookuptable import  lookuptable
from MeanCalculator import MeanCalculator, MeanShift, get_alphas, get_predicted
from GranuleLoader import GranuleLoader
from Utils import save_images, get_root_mean_square

if __name__ == '__main__':
    lut = lookuptable()
    lut.load_flatten_table('../../lookuptable/reflectance/800/800_lookuptable_flatten.numpy')
    data = lut.indices_to_data(lut.flatten_table)

    granule_loader = GranuleLoader()
    granule_loader.bands = [1,2,3,4]
    granule_loader.param = 'reflectance'
    granule_loader.disable_caching()
    granule_loader.disable_multithreading()

    granule_path = '/DATA_5/SNOW_CLOUD_MODIS/data/MOD021KM.A2002179.1640.005.2010085164818.hdf'
    granule_loader.load_granules(granules = [granule_path,])
    original = granule_loader.granules[0].data


    mean_calculator = MeanCalculator()
    mean_calculator.enable_multithreading()
    mean_calculator.enable_caching()

    mean_calculator.threshold = 1e-05
    mean_calculator.number_of_groups = 4
    mean_calculator.number_of_sub_groups = 4
    mean_calculator.number_of_runs = 10
    mean_calculator.clustering_function = "kmeans2"

    means, labels = mean_calculator.calculate_means_data(data)

    alphas = get_alphas(data = data, means = means, labels = labels, training_band = [0,1,2], predictive_band = [3], enable_multithreading = False)
    predicted = get_predicted(data = original, means = means, alphas = alphas, training_band = [0,1,2], predicting_band = [3], enable_multithreading = False)
    save_images(original = original, predicted = predicted, granule_path = granule_path, original_shape = granule_loader.granules[0].original_shape)
    error = get_root_mean_square(original = original[:, 3], predicted = predicted[:, 3])
    print "RMSE: %f%%" % error













