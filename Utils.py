__author__ = 'Samy Vilar'

import multiprocessing, pickle, os, time
from matplotlib import pyplot as plt
import numpy

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


    if red_index and blue_index and green_index:
        source = numpy.dstack((source[crop_origin[0]:crop_origin[0] + crop_size[0], crop_origin[1]:crop_origin[1] + crop_size[1], red_index],
                               source[crop_origin[0]:crop_origin[0] + crop_size[0], crop_origin[1]:crop_origin[1] + crop_size[1], blue_index],
                               source[crop_origin[0]:crop_origin[0] + crop_size[0], crop_origin[1]:crop_origin[1] + crop_size[1], green_index]))
    else:
        source = source[crop_origin[0]:crop_origin[0] + crop_size[0], crop_origin[1]:crop_origin[1] + crop_size[1]]

    plt.imshow(source, vmin = vmin, vmax = vmax, interpolation = interpolation)
    if color_bar:
        plt.colorbar()
    if title:
        plt.suptitle(title)

    if file_name:
        plt.savefig(file_name)






