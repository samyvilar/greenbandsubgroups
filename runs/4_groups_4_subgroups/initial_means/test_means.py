__author__ = 'Samy Vilar'
__date__ = 'Mar 22, 2012'

import sys
sys.path.extend('../../..')

import numpy
import time
from lookuptable.lookuptable import  lookuptable
from MeanCalculator import  minimize, get_predicted_from_means
from Utils import get_sum_of_errors_squared,\
    get_granule_path, get_standard_granule_loader, get_standard_mean_calculator,\
    get_previous_means, save_optimal_solutions
from matplotlib import pyplot as plt


lut = lookuptable()
lut.load_flatten_table('../../lookuptable/reflectance/800_non_capped')
lut_data_flatten = lut.indices_to_data(lut.flatten_table)

granule_path = get_granule_path() + 'MOD021KM.A2002179.1640.005.2010085164818.hdf'

granule_loader = get_standard_granule_loader()
granule_loader.load_granules(granules = [granule_path,])
original = granule_loader.granules[0].data

mean_calculator = get_standard_mean_calculator(multithreading = True,
        caching = False,
        threshold = 1e-09,
        number_of_groups = 4,
        number_of_sub_groups = 4,
        number_of_runs = 20,
        clustering_function = 'kmeans2')

means, labels = mean_calculator.calculate_means_data(lut_data_flatten)



predicted = get_predicted_from_means(data = lut_data_flatten,
                                     means = means,
                                     original = original,
                                     training_band = [0,1,2],
                                     predictive_band = [3],
                                     enable_multithreading = False)

save_images(original = original,
                predicted = predicted,
                granule_path = granule_path,
                original_shape = granule_loader.granules[0].original_shape)

error = get_root_mean_square(original = original[:, 3], predicted = predicted[:, 3])
print "RMSE: %f%%" % error













