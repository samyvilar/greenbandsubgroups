import gc
gc.enable()

import sys
import numpy
sys.path.extend('../../..')

from lookuptable.lookuptable import build_lookuptable, lookuptable
from GranuleLoader import GranuleLoader
from Utils import get_granule_path


if __name__ == '__main__':
    granule_loader = GranuleLoader()
    granule_loader.bands = [1,2,3,4]
    granule_loader.param = 'reflectance'
    granule_loader.disable_caching()
    granule_loader.enable_multithreading()

    '''
    granule_loader_chunks = granule_loader.load_granules_chunk(dir = '/DATA_11/TERRA_1KM', pattern = '*.hdf', chunks = 1)
    max_value = granule_loader_chunks.next()[0].data.max()
    for index, granule in enumerate(granule_loader_chunks):
        if granule and max_value < granule[0].data.max():
            max_value = granule[0].data.max()
            print "new max_value %f" % max_value
        elif granule:
            print "max %f" % granule[0].data.max()
    '''
    max_value = 3.2767000198364257812500000
    granule_loader_chunks = granule_loader.load_granules_chunk(dir = get_granule_path(), pattern = '*.hdf', chunks = 1)
    lut_size = 800

    sums = numpy.zeros((lut_size, lut_size, lut_size))
    counts = numpy.zeros((lut_size, lut_size, lut_size))
    for index, granule in enumerate(granule_loader_chunks):
        if granule:
            try:
                new_lut = build_lookuptable({'data':granule[0].data, 'size':lut_size})
            except Exception as ex:
                continue
            sums += new_lut.sums
            counts += new_lut.counts
            del new_lut
            gc.collect()
        else:
            continue

    print 'done summing and counting ...'
    table = numpy.zeros((lut_size, lut_size, lut_size))
    loc = counts != 0
    table[loc] = sums[loc]/counts[loc]
    table.tofile(str(lut_size) + '_lookuptable_max_value_%.25f_.numpy' % max_value)
    counts.tofile(str(lut_size) + '_counts.numpy')
    sums.tofile(str(lut_size) + '_sums.numpy')

    lut = lookuptable()
    lut.table = table
    lut.size = lut_size
    flatten_table = lut.flatten_2d_non_zero()
    flatten_table.tofile(str(lut_size) + '_lookuptable_flatten_max_value_%.25f_.numpy' % max_value)





