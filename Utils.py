__author__ = 'Samy Vilar'

import multiprocessing, pickle, os, time


class GranuleProperties(object):
    def __init__(self):
        self._bands     = None
        self._param     = None
        self._crop_size = None
        self._crop_orig = None

    @property
    def param(self):
        return self._param
    @property
    def bands(self):
        return self._bands
    @property
    def crop_size(self):
        return self._crop_size
    @property
    def crop_orig(self):
        return self._crop_orig

def get_cpu_count():
    return (multiprocessing.cpu_count() / 4) + multiprocessing.cpu_count()


def load_cached_or_calculate_and_cached(file, func, *args):
    if os.path.isfile(file):
        return pickle.load(file)
    else:
        data = func(args)
        pickle.dump(data, file)
        return data


def multithreading_pool_map(data_array, func):
    pools = multiprocessing.Pool(processes = get_cpu_count())
    start = time.time()
    data = pools.map(func, data_array)
    pools.close()
    pools.join()
    end = time.time()
    print "Done %f" % (end - start)
    return data

