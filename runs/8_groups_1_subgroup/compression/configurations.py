import sys
sys.path.extend('../../..')

import numpy
from Utils import save_images, get_root_mean_square
from matplotlib import pyplot as plt

import numpy
from scipy.io.netcdf import NetCDFFile
from scipy.cluster.vq import kmeans2
from MeanCalculator import get_alphas, get_predicted
from Utils import save_images, get_root_mean_square, get_sum_of_errors_squared, image_show, save_images
from multiprocessing import Pool

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
        [NetCDFFile(cloudy_bands_dir + file).variables[variable_names[index]].data
            for index, file in enumerate(cloudy_bands_files)]).transpose()
cloudy_bands[numpy.isnan(cloudy_bands)] = 0

original_shape = cloudy_bands.shape
cloudy_bands = cloudy_bands.reshape((1185336, 7))

clear_bands = numpy.asarray(
        [NetCDFFile(clear_bands_dir + file).variables[variable_names[index]].data
            for index, file in enumerate(clear_bands_files)]).transpose()
clear_bands[numpy.isnan(clear_bands)] = 0
clear_bands = clear_bands.reshape((1185336, 7))


def get_entropy(values, type = 'uint64'):
    values = values.copy()
    values.dtype = type
    probabilities = {}
    for value in values:
        probabilities[value] = 1 if value not in probabilities \
            else probabilities[value] + 1
    probabilities = numpy.asarray(probabilities.values())/float(len(values))
    return -1*numpy.sum(probabilities * numpy.log2(probabilities))

types = [('uint8', 8), ('uint16', 16), ('uint32', 32), ('uint64', 64)]
cloudy_entropy = numpy.asarray([[get_entropy(cloudy_bands[:, band], type = type[0])
                             for type in types] for band in range(7)])
clear_entropy = numpy.asarray([[ get_entropy(clear_bands[:, band], type = type[0])
                            for type in types] for band in range(7)])

plt.figure()
for index, type in enumerate(types):
    plt.plot(range(1, 8), (cloudy_entropy[:, index])/type[1], label = '%s' % type[0])
plt.xlabel('Band')
plt.ylabel('entropy/word_size')
plt.title('CLOUDY')
plt.legend()
plt.savefig('cloudy_entropy.png')


plt.figure()
for index, type in enumerate(types):
    plt.plot(range(1, 8),  ( clear_entropy[:, index])/type[1], label = '%s' % type[0])
plt.xlabel('Band')
plt.ylabel('entropy/word_size')
plt.title('CLEAR')
plt.legend()
plt.savefig('clear_entropy.png')


def predict(bands, training_bands = [0,1,2,4,5,6], predictive_band = [3], number_of_groups = 20):
    means, labels = kmeans2(bands, number_of_groups)
    alphas = get_alphas(data = bands,
        means = means,
        labels = labels,
        training_band = training_bands,
        predictive_band = predictive_band,
        enable_multithreading = False)
    predicted = get_predicted(
        data = numpy.asarray(bands, dtype='float64', order = 'C'),
        means = means,
        alphas = alphas,
        training_band = training_bands,
        predicting_band = predictive_band,
        enable_multithreading = False)
    return means, predicted, bands[:, predictive_band[0]] - predicted[:, predictive_band[0]]

means, predicted, error = predict(cloudy_bands)
save_images(original = cloudy_bands,
    predicted = predicted,
    granule_path = 'cloudy_bands.nc',
    original_shape = original_shape)


means, predicted, error = predict(clear_bands)
save_images(original = clear_bands,
    predicted = predicted,
    granule_path = 'clear_bands.nc',
    original_shape = original_shape)


all_errors_cloudy = []
all_errors_clear = []

testing_bands = [{'training_bands':range(0, index_1) + range(index_1 + 1, 7),
                  'predicting_bands':[index_1]} for index_1 in xrange(7)]
def get_errors(number_of_groups):
    cloudy_means, cloudy_labels = kmeans2(cloudy_bands, number_of_groups)
    clear_means, clear_labels = kmeans2(clear_bands, number_of_groups)
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
    plt.figure()
    plt.plot(range(1, 8), errors_cloudy, label = 'CLOUDY')
    plt.plot(range(1, 8), errors_clear, label = 'CLEAR')
    plt.legend()
    plt.xlabel('Predicting Band')
    plt.ylabel('Root Mean Square')
    plt.title('Root Errors for %i clusters across different bands.' % number_of_groups)
    plt.savefig('errors_plot_%i_clusters.png' % number_of_groups)
    return numpy.asarray(errors_cloudy), numpy.asarray(errors_clear)



groups = range(4, 22)
pool = Pool(processes = 10)
all_errors = numpy.asarray(pool.map(get_errors, groups))
pool.close()
pool.join()

plt.figure()
plt.plot(groups, numpy.sum(all_errors[:, 0], axis = 1), label = 'CLOUDY')
plt.plot(groups, numpy.sum(all_errors[:, 1], axis = 1), label = 'CLEAR')
plt.legend()
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Root Mean Squares')
plt.title('Number of clusters vs sum of root mean squares')
plt.savefig('Clusters vs sum of errors.png')

plt.figure()
for group in groups:
    plt.plot(range(1, 8), all_errors[group - 4, 0, :], label = 'groups %i' % group)
    plt.xlabel('Predicting Band')
    plt.ylabel('Sum Root Mean Square')
    plt.title('CLOUDY Predicting Band vs root means square')
plt.legend(loc = (1.03,0.2))
plt.savefig('cloudy_groups_errors.png')

plt.figure()
for group in groups:
    plt.plot(range(1, 8), all_errors[group - 4, 1, :], label = 'group %i' % group)
    plt.xlabel('Predicting Band')
    plt.ylabel('Sum Root Mean Square')
    plt.title('CLEAR Predicting Band vs root means square')
plt.legend(loc = (1.03,0.2))
plt.savefig('clear_groups_errors.png')










