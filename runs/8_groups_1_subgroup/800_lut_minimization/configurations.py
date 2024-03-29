
import sys
sys.path.extend('../../..')

import os.path
import pickle
import socket

import numpy
import time
from lookuptable.lookuptable import  lookuptable
from MeanCalculator import MeanCalculator, minimize, get_predicted_from_means
from GranuleLoader import GranuleLoader
from Utils import save_images, get_root_mean_square, get_sum_of_errors_squared
from matplotlib import pyplot as plt

if __name__ == '__main__':
    lut = lookuptable()
    lut.load_flatten_table('../../lookuptable/reflectance/800/800_lookuptable_flatten.numpy')
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

    mean_calculator = MeanCalculator()
    mean_calculator.enable_multithreading()
    mean_calculator.enable_caching()

    mean_calculator.threshold = 1e-05
    mean_calculator.number_of_groups = 8
    mean_calculator.number_of_sub_groups = 1
    mean_calculator.number_of_runs = 10
    mean_calculator.clustering_function = "kmeans2"

    if os.path.isfile('initial_mean.numpy'):
        if os.path.isfile('all_means.obj'):
            all_means = pickle.load(open('all_means.obj', 'rb'))
            sum_of_errors = pickle.load(open('sum_of_errors.obj', 'rb'))
            means = all_means[numpy.asarray(sum_of_errors).argmin()]
        else:
            means = numpy.fromfile('initial_mean.numpy')
            all_means = []
            sum_of_errors = []
    else:
        means, labels = mean_calculator.calculate_means_data(lut_data_flatten)
        means.tofile('initial_mean.numpy')
        all_means = []
        sum_of_errors = []


    training_band = [0,1,2]
    predictive_band = [3]

    output = open('minimization.out', 'a')
    def minimization_function(means):
        start = time.time()
        means  = means.reshape(mean_calculator.number_of_groups, means.shape[0]/mean_calculator.number_of_groups)
        all_means.append(means)
        predicted = get_predicted_from_means(data = lut_data_flatten,
                                             means = means,
                                             original = test_data,
                                             training_band = training_band,
                                             predictive_band = predictive_band,
                                             enable_multithreading = False)
        sum_of_errors.append(get_sum_of_errors_squared(
                predicted = predicted[:, predictive_band[0]],
                original  = test_data[:, predictive_band[0]]))
        pickle.dump(all_means, open('all_means.obj', 'wb'))
        pickle.dump(sum_of_errors, open('sum_of_errors.obj', 'wb'))
        output.write("minimization_function iteration: " + str(len(sum_of_errors)) + " time to finnish: " + str(round((time.time() - start), 8)) + "s Sum Of Error: " + str(sum_of_errors[-1]) + '\n')
        output.flush()
        return sum_of_errors[-1]

    opt_means = minimize(initial_values = means,
                         function = minimization_function,
                         max_iterations = 1000)
    pickle.dump(opt_means, open('opt_means.obj', 'ab'))
    pickle.dump(sum_of_errors, open('sum_of_errors.obj', 'ab'))

    predicted = get_predicted_from_means(data = lut_data_flatten,
                                            means = opt_means,
                                            original = test_data,
                                            training_band = training_band,
                                            predictive_band = predictive_band,
                                            enable_multithreading = False)

    save_images(original = test_data,
                predicted = predicted,
                granule_path = granule_path,
                original_shape = granule_loader.granules[0].original_shape)

    error = get_root_mean_square(original = test_data[:, predictive_band[0]],
                                 predicted = predicted[:, predictive_band[0]])


    plt.plot(sum_of_errors)
    plt.savefigure('sum_of_errors_per_iterations.png')

    print "RMSE: %f%%" % error
