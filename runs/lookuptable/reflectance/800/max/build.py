import gc
gc.enable()

import sys
import numpy
sys.path.extend('../../../..')

from lookuptable.lookuptable import build_lookuptable, lookuptable, update_max
from Utils import get_all_granules_path, get_standard_granule_loader

if __name__ == '__main__':
    granule_loader = get_standard_granule_loader()
    chunk_size = 1
    granule_loader_chunks = granule_loader.load_granules_chunk(dir = get_all_granules_path(), pattern = '*.hdf', chunks = chunk_size)
    lut_size = 800
    max_value = 1
    max = None

    for index, granule in enumerate(granule_loader_chunks):
        if granule:
            try:
                new_lut = build_lookuptable({'data':granule[0].data,
                                             'size':lut_size,
                                             'max_value':max_value,
                                             'function':'max'})
            except Exception as ex:
                print str(ex)
                continue
            if max == None:
                max = new_lut.max
            else:
                update_max(prev_max = max, new_max = max, lut_size = lut_size)
            del new_lut
            gc.collect()
        else:
            continue

        max.tofile(str(lut_size) + '_max_lookuptable.numpy')

        lut = lookuptable()
        lut.table = max
        lut.size = lut_size
        flatten_table = lut.flatten_2d_non_zero()
        flatten_table.tofile(str(lut_size) + '_max_lookuptable_flatten.numpy')

'''
2 mat files, one sorted by the (max - min) and the other by the standard deviation
RGB, max - min, max, min, std, count
'''



