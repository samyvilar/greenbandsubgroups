import gc
gc.enable()

import sys
import numpy
sys.path.extend('../../../..')

from lookuptable.lookuptable import update_min_max, flatten_2d_non_zero
from Utils import get_all_granules_path, get_standard_granule_loader

if __name__ == '__main__':
    granule_loader = get_standard_granule_loader()
    chunk_size = 1
    granule_loader_chunks = granule_loader.load_granules_chunk(dir = get_all_granules_path(), pattern = '*.hdf', chunks = chunk_size)
    lut_size = 800
    max_value = 1
    mins = numpy.zeros((lut_size, lut_size, lut_size), dtype = 'float32')
    mins.fill(10000)
    max = numpy.zeros((lut_size, lut_size, lut_size), dtype = 'float32')
    max.fill(-10000)

    for index, granule in enumerate(granule_loader_chunks):
        if granule:
            try:
                update_min_max({'data':granule[0].data,
                                'mins':mins,
                                'max':max,
                                'size':lut_size,
                                'max_value':max_value,
                                })
                print 'Granule: %s' % granule[0].file_name
            except Exception as ex:
                print 'Exception: ' + str(ex)
                continue

    mins = mins.astype('float64')
    max = max.astype('float64')

    mins.tofile(str(lut_size) + '_min_lookuptable.numpy')
    max.tofile(str(lut_size) + '_max_lookuptable.numpy')



    flatten_table = flatten_2d_non_zero(table = mins, default_value = 10000, size = lut_size)
    flatten_table.tofile(str(lut_size) + '_min_lookuptable_flatten.numpy')

    flatten_table = flatten_2d_non_zero(table = max, default_value = -10000, size = lut_size)
    flatten_table.tofile(str(lut_size) + '_max_lookuptable_flatten.numpy')
