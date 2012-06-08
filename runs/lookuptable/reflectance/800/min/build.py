import gc
gc.enable()

import sys
import numpy
sys.path.extend('../../../..')

from lookuptable.lookuptable import build_lookuptable, lookuptable
from Utils import get_all_granules_path, get_standard_granule_loader

if __name__ == '__main__':
    granule_loader = get_standard_granule_loader()
    #/DATA_11/TERRA_1KM/
    chunk_size = 1
    granule_loader_chunks = granule_loader.load_granules_chunk(dir = get_all_granules_path(), pattern = '*.hdf', chunks = chunk_size)
    lut_size = 800
    max_value = 1

    mins = None
    for index, granule in enumerate(granule_loader_chunks):
        if granule:
            try:
                new_lut = build_lookuptable({'data':granule[0].data,
                                             'size':lut_size,
                                             'max_value':max_value,
                                             'function':'min'})
            except Exception as ex:
                print str(ex)
                continue
            if mins == None:
                mins = new_lut.min
            else:
                mins = numpy.asarray([mins, new_lut.mins]).min(axis = 0)
            del new_lut
            gc.collect()
        else:
            continue

    mins[mins == (max_value + 1)] = numpy.nan
    mins.tofile(str(lut_size) + '_min_lookuptable.numpy')

    lut = lookuptable()
    lut.table = mins
    lut.size = lut_size
    flatten_table = lut.flatten_2d_non_zero()
    flatten_table.tofile(str(lut_size) + '_min_lookuptable_flatten.numpy')





