import gc
gc.enable()

import sys
import numpy
sys.path.extend('../../..')

from lookuptable.lookuptable import lookuptable, build_lookuptable
from GranuleLoader import GranuleLoader
from Utils import multithreading_pool_map

#if __name__ == '__main__':
lut = lookuptable()
granule_loader = GranuleLoader()
granule_loader.bands = [1,2,3,4]
granule_loader.param = 'reflectance'
granule_loader.disable_caching()
granule_loader.enable_multithreading()

granule_loader_chunks = granule_loader.load_granules_chunk(dir = '/DATA_11/TERRA_1KM', pattern = '*.hdf', chunks = 1)
lut_size = 1000

lut = build_lookuptable({'data':granule_loader_chunks.next()[0].data, 'size':lut_size})

granule = granule_loader_chunks.next()[0]
temp_lut = build_lookuptable({'data':granule[0].data, 'size':lut_size})
prev_counts = numpy.copy(lut.counts)
lut.counts += temp_lut.counts
non_zero_locations = lut.counts != 0
lut.table = (temp_lut.sums[non_zero_locations]/lut.counts[non_zero_locations]) + (prev_counts * lut.table)/lut.counts[non_zero_locations]
lut.table[non_zero_locations] /= lut.table[lut.counts != 0]

#tables = multithreading_pool_map(function = build_lookuptable, values = [{'data':c.data, 'size':600} for c in chunk], multithreaded = True)



