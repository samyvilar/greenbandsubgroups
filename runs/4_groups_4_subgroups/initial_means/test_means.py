__author__ = 'Samy Vilar'
__date__ = 'Mar 22, 2012'

import sys
sys.path.extend('../../..')

import os.path
import pickle

import numpy
import time
from lookuptable.lookuptable import  lookuptable
from MeanCalculator import MeanCalculator, minimize, get_predicted_from_means
from GranuleLoader import GranuleLoader
from Utils import save_images, get_root_mean_square, get_sum_of_errors_squared, get_granule_path
from matplotlib import pyplot as plt

lut = lookuptable()
lut.load_flatten_table('../../lookuptable/reflectance/800_non_capped')
lut_data_flatten = lut.indices_to_data(lut.flatten_table)

granule_loader = GranuleLoader()
granule_loader.bands = [1,2,3,4]
granule_loader.param = 'reflectance'
granule_loader.disable_caching()
granule_loader.disable_multithreading()
granule_path = get_granule_path() + 'MOD021KM.A2002179.1640.005.2010085164818.hdf'

granule_loader.load_granules(granules = [granule_path,])
original = granule_loader.granules[0].data

mean_calculator = MeanCalculator()
mean_calculator.enable_multithreading()
mean_calculator.disable_caching()

mean_calculator.threshold = 1e-05
mean_calculator.number_of_groups = 2
mean_calculator.number_of_sub_groups = 2
mean_calculator.number_of_runs = 10
mean_calculator.clustering_function = "kmeans2"

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













