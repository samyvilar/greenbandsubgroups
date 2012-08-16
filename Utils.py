__author__ = 'Samy Vilar' 

import multiprocessing, pickle, os, time
from matplotlib import pyplot as plt
import numpy
from os.path import basename
import socket
from GranuleLoader import GranuleLoader
from MeanCalculator import MeanCalculator, get_predicted_from_means

def get_cpu_count():
    return (multiprocessing.cpu_count() / 4) + multiprocessing.cpu_count()

def load_cached_or_calculate_and_cached(caching = None, file_name = None, function = None, arguments = None):
    if not caching:
        return function(**arguments)

    if os.path.isfile(file_name):
        return pickle.load(open(file_name, 'rb'))
    else:
        if '/' in file_name and not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))

        values = function(**arguments)
        pickle.dump(values, open(file_name, 'wb'))
        return values


def multithreading_pool_map(**kwargs):
    values          = kwargs.pop('values')
    function        = kwargs.pop('function')
    multithreaded   = kwargs.pop('multithreaded')

    if multithreaded:
        pools = multiprocessing.Pool(processes = get_cpu_count())
        start = time.time()
        results = pools.map(function, values)
        pools.close()
        pools.join()
        end = time.time()
        print "Done %fs" % (end - start)
        return results

    start = time.time()
    results = [function(value) for value in values]
    end = time.time()
    print "Done %fs" % (end - start)
    return results

def image_show(**kwargs):
    source          = kwargs.pop('source',          None)
    vmin            = kwargs.pop('vmin',            None)
    vmax            = kwargs.pop('vmax',            None)
    min             = kwargs.pop('min',             None)
    max             = kwargs.pop('max',             None)
    reshape         = kwargs.pop('reshape',         None)
    file_name       = kwargs.pop('file_name',       None)
    color_bar       = kwargs.pop('color_bar',       None)
    interpolation   = kwargs.pop('interpolation',   None)
    title           = kwargs.pop('title',           None)
    red_index       = kwargs.pop('red_index',       None)
    blue_index      = kwargs.pop('blue_index',      None)
    green_index     = kwargs.pop('green_index',     None)
    crop_origin     = kwargs.pop('crop_origin',     (0,0))
    crop_size       = kwargs.pop('crop_size',       None)
    dir             = kwargs.pop('dir',             '')

    if kwargs:
        raise ValueError('Received invalid keyword argument(s) %s' % str(kwargs))

    assert source is not None

    plt.figure()
    if reshape:
        source = source.reshape(reshape)
    if min:
        source[source < min] = min
    if max:
        source[source > max] = max

    if not crop_size:
        crop_size = source.shape

    if red_index is not None and blue_index is not None and green_index is not None:
        source = numpy.dstack((source[crop_origin[0]:(crop_origin[0] + crop_size[0]), crop_origin[1]:(crop_origin[1] + crop_size[1]), red_index],
                               source[crop_origin[0]:(crop_origin[0] + crop_size[0]), crop_origin[1]:(crop_origin[1] + crop_size[1]), green_index],
                               source[crop_origin[0]:(crop_origin[0] + crop_size[0]), crop_origin[1]:(crop_origin[1] + crop_size[1]), blue_index]))
    else:
        source = source[crop_origin[0]:crop_origin[0] + crop_size[0], crop_origin[1]:crop_origin[1] + crop_size[1]]

    plt.imshow(source, vmin = vmin, vmax = vmax, interpolation = interpolation)
    if color_bar:
        plt.colorbar()
    if title:
        plt.suptitle(title)

    dir = dir + '/' if dir and dir[-1] != '/' else dir
    if file_name:
            plt.savefig(dir + file_name)



def get_root_mean_square(original = None, predicted = None):
    assert original is not None and predicted is not None
    std = numpy.sqrt(numpy.sum((predicted - original)**2)/original.shape[0])
    return std/numpy.mean(original) * 100

def get_sum_of_errors_squared(original = None, predicted = None):
    assert original is not None and predicted is not None
    return numpy.sum(numpy.square(original - predicted))

def save_images(**kwargs):
    original        = kwargs['original']
    predicted       = kwargs['predicted']
    granule_path    = kwargs['granule_path']
    original_shape  = kwargs['original_shape']
    dir             = kwargs.get('dir', '')

    original_green = original[:, 3].reshape(original_shape[0:2])
    predicted_green = predicted[:, 3].reshape(original_shape[0:2])
    image_show(source = original_green,
        vmin = 0, vmax = 1, min = 0, max = 1,
        interpolation = 'nearest',
        color_bar = True, file_name = 'original_green.png',
        title = 'True Green Values granule %s' % basename(granule_path),
        dir = dir)
    image_show(source = predicted_green,
        vmin = 0, vmax = 1, min = 0, max = 1,
        interpolation = 'nearest',
        color_bar = True, file_name = 'predicted_green.png',
        title = 'Predicted Green Values granule %s' % basename(granule_path),
        dir = dir)
    image_show(source = numpy.fabs(original_green - predicted_green),
        vmin = 0, vmax = 1, min = 0, max = 1,
        interpolation = 'nearest',
        color_bar = True, file_name = 'error.png',
        title = 'Absolute Error of Green Values Predicted vs True',
        dir = dir)
    image_show(source = numpy.fabs(original_green - predicted_green)/original_green,
        vmin = 0, vmax = 1, min = 0, max = 1,
        interpolation = 'nearest',
        color_bar = True, file_name = 'relative_error.png',
        title = 'Relative Error of Green Values Predicted vs True',
        dir = dir)
    image_show(source = numpy.fabs(original_green - predicted_green)/original_green,
        vmin = 0, vmax = .25, min = 0, max = 1,
        interpolation = 'nearest',
        color_bar = True, file_name = 'relative_error_cropped.png',
        crop_origin = (0, 0), crop_size = (1000, 1000),
        title = 'LUT Relative Error of Green Values Predicted vs True CROPPED',
        dir = dir)
    image_show(
        source = original,
        reshape = original_shape,
        vmin = 0, vmax = 1, min = 0, max = 1,
        interpolation = 'nearest',
        color_bar = True, file_name = 'original_rgb_type_casted.png',
        title = 'True RGB granule %s' % basename(granule_path),
        red_index = 0, green_index = 3, blue_index = 2,
        dir = dir)
    image_show(source = predicted,
        reshape = original_shape,
        vmin = 0, vmax = 1,
        interpolation = 'nearest',
        color_bar = True, file_name = 'predicted_rgb_type_casted',
        title = 'Predicted RGB granule %s' % basename(granule_path),
        red_index = 0, green_index = 3, blue_index = 2,
        dir = dir)
    image_show(source = original,
        reshape = original_shape,
        vmin = 0, vmax = 1, min = 0, max = 1,
        interpolation = 'nearest',
        crop_origin = (500, 0), crop_size = (750, 600),
        color_bar = True, file_name = 'original_rgb_type_casted_500-1250_0-600.png',
        title = 'True RGB type casted and cropped granule %s' % basename(granule_path),
        red_index = 0, green_index = 3, blue_index = 2,
        dir = dir)
    image_show(source = predicted,
        reshape = original_shape,
        vmin = 0, vmax = 1,
        interpolation = 'nearest',
        crop_origin = (500, 0), crop_size = (750, 600),
        color_bar = True, file_name = 'predicted_rgb_type_casted_500-1250_0-600.png',
        title = 'Predicted RGB type casted and cropped granule %s' % basename(granule_path),
        red_index = 0, green_index = 3, blue_index = 2,
        dir = dir)


def get_granule_path():
    return '/DATA_5/TERRA/' if 'fermi' in socket.gethostname() else '/home1/FermiData/DATA_5/SNOW_CLOUD_MODIS/data/'

def get_all_granules_path():
    return '/DATA_11/TERRA_1KM/' if 'foucault' in socket.gethostname() else '/home1/FoucaultData/DATA_11/TERRA_1KM/'

def get_standard_granule_loader(bands = [1,2,3,4], parameter = 'reflectance', caching = False, multithreading = False):
    granule_loader = GranuleLoader()
    granule_loader.bands = bands
    granule_loader.param = parameter
    if caching:
        granule_loader.enable_caching()
    else:
        granule_loader.disable_caching()
    if multithreading:
        granule_loader.enable_multithreading()
    else:
        granule_loader.disable_multithreading()
    return granule_loader

def get_standard_mean_calculator(threshold = None,
                                 number_of_groups = None,
                                 number_of_sub_groups = None,
                                 number_of_runs = None,
                                 clustering_function = None,
                                 multithreading = True,
                                 caching = False):
    assert threshold and number_of_groups and number_of_sub_groups and number_of_runs and clustering_function
    mean_calculator = MeanCalculator()

    if multithreading:
        mean_calculator.enable_multithreading()
    else:
        mean_calculator.disable_caching()
    if caching:
        mean_calculator.enable_caching()
    else:
        mean_calculator.disable_caching()

    mean_calculator.threshold = threshold
    mean_calculator.number_of_groups = number_of_groups
    mean_calculator.number_of_sub_groups = number_of_sub_groups
    mean_calculator.number_of_runs = number_of_runs
    mean_calculator.clustering_function = clustering_function
    return mean_calculator

def get_previous_means(mean_calculator = None, lut_data_flatten = None):
    assert mean_calculator is not None
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
    return all_means, sum_of_errors, means

def save_optimal_solutions(**kwargs):
    opt_means = numpy.copy(kwargs['opt_means'])
    dir = kwargs['dir']
    lut_data_flatten = numpy.copy(kwargs['lut_data_flatten'])
    original = numpy.copy(kwargs['original'])
    training_band = kwargs['training_band']
    predictive_band = kwargs['predictive_band']
    granule_path = kwargs['granule_path']
    original_shape = kwargs['original_shape']
    sum_of_errors = numpy.copy(kwargs['sum_of_errors'])

    pickle.dump(opt_means, open('%s/%s' % (dir, 'opt_means.obj'), 'wb'))
    predicted = get_predicted_from_means(data = lut_data_flatten,
        means = opt_means,
        original = original,
        training_band = training_band,
        predictive_band = predictive_band,
        enable_multithreading = False)
    save_images(original = original,
        predicted = predicted,
        granule_path = granule_path,
        original_shape = original_shape,
        dir = dir)
    error = get_root_mean_square(original = original[:, predictive_band[0]],
        predicted = predicted[:, predictive_band[0]])
    plt.figure()
    plt.plot(sum_of_errors)
    plt.savefig(dir + '/sum_of_errors_per_iterations.png')
    open(dir + "/RMSE:%f%%" % error, 'wb')

