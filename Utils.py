__author__ = 'Samy Vilar'

import multiprocessing, pickle, os, time
from matplotlib import pyplot as plt
import numpy
from os.path import basename

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
        print "Done %f" % (end - start)
        return results

    start = time.time()
    results = [function(value) for value in values]
    end = time.time()
    print "Done %f" % (end - start)
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

    if kwargs:
        raise ValueError('Received invalid keyword argument(s) %s' % str(kwargs))

    assert source != None

    plt.figure()
    if reshape:
        source = source.reshape(reshape)
    if min:
        source[source < min] = min
    if max:
        source[source > max] = max

    if not crop_size:
        crop_size = source.shape

    if red_index != None and blue_index != None and green_index != None:
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

    if file_name:
        plt.savefig(file_name)


def get_root_mean_square(original = None, predicted = None):
    assert original != None and predicted != None
    std = numpy.sqrt(numpy.sum((predicted - original)**2)/original.shape[0])
    return std/numpy.mean(original) * 100

def save_images(**kwargs):
    original        = kwargs['original']
    predicted       = kwargs['predicted']
    granule_path    = kwargs['granule_path']
    original_shape  = kwargs['original_shape']

    original_green = original[:, 3].reshape(original_shape[0:2])
    predicted_green = predicted[:, 3].reshape(original_shape[0:2])
    image_show(source = original_green,
        vmin = 0, vmax = 1, min = 0, max = 1,
        interpolation = 'nearest',
        color_bar = True, file_name = 'original_green.png',
        title = 'True Green Values granule %s' % basename(granule_path))
    image_show(source = predicted_green,
        vmin = 0, vmax = 1, min = 0, max = 1,
        interpolation = 'nearest',
        color_bar = True, file_name = 'predicted_green.png',
        title = 'Predicted Green Values granule %s' % basename(granule_path))
    image_show(source = numpy.fabs(original_green - predicted_green),
        vmin = 0, vmax = 1, min = 0, max = 1,
        interpolation = 'nearest',
        color_bar = True, file_name = 'error.png',
        title = 'Absolute Error of Green Values Predicted vs True')
    image_show(source = numpy.fabs(original_green - predicted_green)/original_green,
        vmin = 0, vmax = 1, min = 0, max = 1,
        interpolation = 'nearest',
        color_bar = True, file_name = 'relative_error.png',
        title = 'Relative Error of Green Values Predicted vs True')
    image_show(source = numpy.fabs(original_green - predicted_green)/original_green,
        vmin = 0, vmax = .25, min = 0, max = 1,
        interpolation = 'nearest',
        color_bar = True, file_name = 'relative_error_cropped.png',
        crop_origin = (0, 0), crop_size = (1000, 1000),
        title = 'LUT Relative Error of Green Values Predicted vs True CROPPED')
    image_show(
        source = original,
        reshape = original_shape,
        vmin = 0, vmax = 1, min = 0, max = 1,
        interpolation = 'nearest',
        color_bar = True, file_name = 'original_rgb_type_casted.png',
        title = 'True RGB granule %s' % basename(granule_path),
        red_index = 0, green_index = 3, blue_index = 2)
    image_show(source = predicted,
        reshape = original_shape,
        vmin = 0, vmax = 1,
        interpolation = 'nearest',
        color_bar = True, file_name = 'predicted_rgb_type_casted',
        title = 'Predicted RGB granule %s' % basename(granule_path),
        red_index = 0, green_index = 3, blue_index = 2)
    image_show(source = original,
        reshape = original_shape,
        vmin = 0, vmax = 1, min = 0, max = 1,
        interpolation = 'nearest',
        crop_origin = (500, 0), crop_size = (750, 600),
        color_bar = True, file_name = 'original_rgb_type_casted_500-1250_0-600.png',
        title = 'True RGB type casted and cropped granule %s' % basename(granule_path),
        red_index = 0, green_index = 3, blue_index = 2)
    image_show(source = predicted,
        reshape = original_shape,
        vmin = 0, vmax = 1,
        interpolation = 'nearest',
        crop_origin = (500, 0), crop_size = (750, 600),
        color_bar = True, file_name = 'predicted_rgb_type_casted_500-1250_0-600.png',
        title = 'Predicted RGB type casted and cropped granule %s' % basename(granule_path),
        red_index = 0, green_index = 3, blue_index = 2)


