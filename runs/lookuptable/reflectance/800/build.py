import gc
gc.enable()

import sys
import numpy
sys.path.extend('../../..')

from lookuptable.lookuptable import build_lookuptable, flatten_2d_non_zero
from GranuleLoader import GranuleLoader
from Utils import get_all_granules_path

if __name__ == '__main__':
    granule_loader = GranuleLoader()
    granule_loader.bands = [1,2,3,4]
    granule_loader.param = 'reflectance'
    granule_loader.disable_caching()
    granule_loader.enable_multithreading()


    #/DATA_11/TERRA_1KM/
    granule_loader_chunks = granule_loader.load_granules_chunk(dir = get_all_granules_path(), pattern = '*.hdf', chunks = 1)
    lut_size = 800
    sums = numpy.zeros((lut_size, lut_size, lut_size), dtype = 'uint64')
    counts = numpy.zeros((lut_size, lut_size, lut_size), dtype = 'uint32')

    for index, granule in enumerate(granule_loader_chunks):
        if granule:
            try:
                result = build_lookuptable({'data':granule[0].data,
                                            'size':lut_size,
                                            'max_value':1,
                                            })

                sums += result['sum']
                counts += result['counts']
            except Exception as ex:
                print 'Exception: ' + str(ex)
                continue

    table = numpy.zeros((lut_size, lut_size, lut_size))
    loc = counts != 0
    table[loc] = sums[loc]/counts[loc]
    table.tofile(str(lut_size) + '_lookuptable.numpy')
    counts.tofile(str(lut_size) + '_counts_uint32.numpy')
    sums.tofile(str(lut_size) + '_sums_uint64.numpy')

    flatten_table = flatten_2d_non_zero(table = table, size = lut_size)
    flatten_table.tofile(str(lut_size) + '_lookuptable_flatten.numpy')
