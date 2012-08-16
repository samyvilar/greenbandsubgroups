import gc
gc.enable()

import sys
import numpy
sys.path.extend('../../../..')

from lookuptable.lookuptable import build_lookuptable, lookuptable, update_min
from Utils import get_all_granules_path, get_standard_granule_loader

if __name__ == '__main__':
    granule_loader = get_standard_granule_loader()
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
                mins = update_min(prev_mins = mins,
                           new_mins = new_lut.min,
                           lut_size = lut_size,
                           method = 'C')
            del new_lut

    mins.tofile(str(lut_size) + '_min_lookuptable.numpy')

    lut = lookuptable()
    lut.table = mins
    lut.size = lut_size
    flatten_table = lut.flatten_2d_non_zero(default_value = 900)
    flatten_table.tofile(str(lut_size) + '_min_lookuptable_flatten.numpy')

    import time
    from matplotlib import pyplot as pltlib

    pltlib.ion()
    pltlib.interactive(True)
    pltlib.figure(1)
    pltlib.plot(range(10),range(10), "r-")
    pltlib.show()

    deconvFig = pltlib.figure(2)
    ax = deconvFig.add_subplot(111)
    X, Y = range(10), range(10)
    line1,line2 = ax.plot(X,Y,'r-',X,Y,'r-')
    for x in xrange(2, 6, 1):
        line2.set_ydata(range(0, 10*x, x))
        deconvFig.canvas.draw()
        time.sleep(2)
