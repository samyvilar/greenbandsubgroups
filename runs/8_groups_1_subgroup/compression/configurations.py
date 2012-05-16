import sys
sys.path.extend('../../..')

import os.path
import pickle
import socket

import numpy
import time
from lookuptable.lookuptable import  lookuptable
from GranuleLoader import GranuleLoader
from Utils import save_images, get_root_mean_square, get_sum_of_errors_squared
from matplotlib import pyplot as plt

import numpy
from scipy.io.netcdf import NetCDFFile
from scipy.cluster.vq import kmeans2
from scipy.spatial import kdtree
from MeanCalculator import get_alphas, get_predicted
from Utils import save_images, get_root_mean_square, get_sum_of_errors_squared

clear_bands_dir, cloudy_bands_dir = 'clear_bands/', 'cloudy_bands/'
variable_names = ['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7']
clear_bands_files = ['MOD02HKM.A2010271.1740.B1.nc',
                     'MOD02HKM.A2010271.1740.B2.nc',
                     'MOD02HKM.A2010271.1740.B3.nc',
                     'MOD02HKM.A2010271.1740.B4.nc',
                     'MOD02HKM.A2010271.1740.B5.nc',
                     'MOD02HKM.A2010271.1740.B6.nc',
                     'MOD02HKM.A2010271.1740.B7.nc']

cloudy_bands_files = ['MOD02HKM.A2010278.1745.B1.nc',
                      'MOD02HKM.A2010278.1745.B2.nc',
                      'MOD02HKM.A2010278.1745.B3.nc',
                      'MOD02HKM.A2010278.1745.B4.nc',
                      'MOD02HKM.A2010278.1745.B5.nc',
                      'MOD02HKM.A2010278.1745.B6.nc',
                      'MOD02HKM.A2010278.1745.B7.nc']

cloudy_bands = numpy.asarray(
        [NetCDFFile(cloudy_bands_dir + file).variables[variable_names[index]].data.reshape((1185336,))
            for index, file in enumerate(cloudy_bands_files)]).transpose()
cloudy_bands[numpy.isnan(cloudy_bands)] = 0

clear_bands = numpy.asarray(
        [NetCDFFile(clear_bands_dir + file).variables[variable_names[index]].data.reshape((1185336,))
            for index, file in enumerate(clear_bands_files)]).transpose()
clear_bands[numpy.isnan(clear_bands)] = 0

all_errors_cloudy = []
all_errors_clear = []
for index in xrange(4, 10):
    cloudy_means, cloudy_labels = kmeans2(cloudy_bands, index)
    clear_means, clear_labels = kmeans2(clear_bands, index)
    testing_bands = [{'training_bands':range(0, index_1) + range(index_1 + 1, index),
                        'predicting_bands':[index]} for index_1 in xrange(index)]
    errors_cloudy = []
    errors_clear = []
    for testing_band in testing_bands:
        alphas = get_alphas(data = cloudy_bands,
                            means = cloudy_means,
                            labels = cloudy_labels,
                            training_band = testing_band['training_bands'],
                            predictive_band = testing_band['predicting_bands'],
                            enable_multithreading = False)
        predicted = get_predicted(data = numpy.asarray(cloudy_bands, dtype='float64', order = 'C'),
                                 means = cloudy_means,
                                 alphas = alphas,
                                 training_band = testing_band['training_bands'],
                                 predicting_band = testing_band['predicting_bands'],
                                 enable_multithreading = False)
        errors_cloudy.append(get_root_mean_square(original = cloudy_bands[:, testing_band['predicting_bands'][0]],
                                predicted = predicted[:, testing_band['predicting_bands'][0]]))
        alphas = get_alphas(data = clear_bands,
                            means = clear_means,
                            labels = clear_labels,
                            training_band = testing_band['training_bands'],
                            predictive_band = testing_band['predicting_bands'],
                            enable_multithreading = False)
        predicted = get_predicted(data = numpy.asarray(clear_bands, dtype='float64', order = 'C'),
                                means = clear_means,
                                alphas = alphas,
                                training_band = testing_band['training_bands'],
                                predicting_band = testing_band['predicting_bands'],
                                enable_multithreading = False)
        errors_clear.append(get_root_mean_square(original = clear_bands[:, testing_band['predicting_bands'][0]],
                        predicted = predicted[:, testing_band['predicting_bands'][0]]))
    all_errors_clear.append(errors_clear)
    all_errors_cloudy.append(errors_cloudy)
    plt.figure()
    cloudy_line = plt.plot(range(1, index + 1), errors_cloudy, label = 'CLOUDY')
    clear_line = plt.plot(range(1, index + 1), errors_clear, label = 'CLEAR')
    plt.legend()
    plt.xlabel('Root Mean Square')
    plt.ylabel('Root Mean Square')
    plt.title('Root Errors for %i clusters across different bands.' % index)
    plt.savefig('errors_plot_%i_clusters.png' % index)





