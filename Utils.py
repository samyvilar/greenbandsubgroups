__author__ = 'Samy Vilar'

import multiprocessing, pickle, os, time

def get_cpu_count():
    return (multiprocessing.cpu_count() / 4) + multiprocessing.cpu_count()


def load_cached_or_calculate_and_cached(caching = None, file_name = None, function = None, arguments = None):
    if not caching:
        return function(arguments)

    if os.path.isfile(file_name):
        return pickle.load(file_name)
    else:
        values = function(**arguments)
        pickle.dump(data, file_name)
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


