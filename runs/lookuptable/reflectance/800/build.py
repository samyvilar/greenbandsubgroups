import gc
gc.enable()

import sys
import numpy
sys.path.extend('../../..')

from lookuptable.lookuptable import build_lookuptable
from GranuleLoader import GranuleLoader
from progressbar import ProgressBar, Percentage, Bar, RotatingMarker, ETA



if __name__ == '__main__':
    granule_loader = GranuleLoader()
    granule_loader.bands = [1,2,3,4]
    granule_loader.param = 'reflectance'
    granule_loader.disable_caching()
    granule_loader.enable_multithreading()

    granule_loader_chunks = granule_loader.load_granules_chunk(dir = '/home1/FoucaultData/DATA_11/TERRA_1KM', pattern = '*.hdf', chunks = 1)
    lut_size = 800

    sums = numpy.zeros((lut_size, lut_size, lut_size))
    counts = numpy.zeros((lut_size, lut_size, lut_size))
    for index, granule in enumerate(granule_loader_chunks):
        try:
            new_lut = build_lookuptable({'data':granule[0].data, 'size':lut_size})
        except Exception as ex:
            continue

        sums += new_lut.sums
        counts += new_lut.counts
        del new_lut
        gc.collect()

    table = numpy.zeros((lut_size, lut_size, lut_size))
    loc = counts != 0
    table[loc] = sums[loc]/counts[loc]
    table.tofile(str(lut_size) + '_lookuptable.numpy')
    counts.tofile(str(lut_size) + '_counts.numpy')
    sums.tofile(str(lut_size) + '_sums.numpy')




