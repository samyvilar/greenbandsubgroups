
import sys
sys.path.extend('../../..')

import os.path
import os
import pickle
import shutil

import numpy
import time
from lookuptable.lookuptable import  lookuptable
from MeanCalculator import  minimize, get_predicted_from_means
from Utils import get_sum_of_errors_squared,\
    get_granule_path, get_standard_granule_loader, get_standard_mean_calculator,\
    get_previous_means, save_optimal_solutions
from matplotlib import pyplot as plt

if __name__ == '__main__':
    lut = lookuptable()
    lut.load_flatten_table('../../lookuptable/reflectance/800_non_capped')
    lut_data_flatten = lut.indices_to_data(lut.flatten_table)

    granule_path = get_granule_path() + 'MOD021KM.A2002179.1640.005.2010085164818.hdf'

    granule_loader = get_standard_granule_loader()
    granule_loader.load_granules(granules = [granule_path,])
    test_data = granule_loader.granules[0].data

    mean_calculator = get_standard_mean_calculator(multithreading = True,
        caching = False,
        threshold = 1e-06,
        number_of_groups = 11,
        number_of_sub_groups = 1,
        number_of_runs = 10,
        clustering_function = 'kmeans2')

    all_means, sum_of_errors, means = get_previous_means(mean_calculator = mean_calculator, lut_data_flatten = lut_data_flatten)

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

        if not (len(sum_of_errors) % 1000):
            dir = 'images_%i_iterations' % len(sum_of_errors)
            os.makedirs(dir)
            files = ['all_means.obj', 'initial_mean.numpy', 'minimization.out', 'nohup.out', 'sum_of_errors.obj']
            for file in files:
                if os.path.isfile(file):
                    shutil.copy(file, dir + '/')
            opt_means = all_means[numpy.asarray(sum_of_errors).argmin()]
            save_optimal_solutions(dir = dir, opt_means = opt_means, lut_data_flatten = lut_data_flatten,
                original = test_data, training_band = training_band, predictive_band = predictive_band,
                granule_path = granule_path, original_shape = granule_loader.granules[0].original_shape,
                sum_of_errors = sum_of_errors)





        return sum_of_errors[-1]

    minimize(initial_values = means, function = minimization_function)

    opt_means = all_means[numpy.asarray(sum_of_errors).argmin()]
    save_optimal_solutions(dir = dir, opt_means = opt_means, lut_data_flatten = lut_data_flatten,
        original = test_data, training_band = training_band, predictive_band = predictive_band,
        granule_path = granule_path, original_shape = granule_loader.granules[0].original_shape,
        sum_of_errors = sum_of_errors)
