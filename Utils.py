__author__ = 'Samy Vilar'

import multiprocessing, pickle, os, time

def get_cpu_count():
    return (multiprocessing.cpu_count() / 4) + multiprocessing.cpu_count()


def load_cached_or_calculate_and_cached(caching = None, file_name = None, function = None, arguments = None):
    if not caching:
        return function(**arguments)

    if os.path.isfile(file_name):
        return pickle.load(open(file_name, 'rb'))
    else:
        if '/' in file_name and not os.path.exists(file_name.split('/')[:-1]):
            os.makedirs(''.join(['/' + name for name in file_name.split('/')[:-1]]))

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
