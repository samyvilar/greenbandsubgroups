import gc
gc.enable()

import sys
import numpy
sys.path.extend('../../..')

from lookuptable.lookuptable import lookuptable, build_lookuptable
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

    #lut = build_lookuptable({'data':granule_loader_chunks.next()[0].data, 'size':lut_size})

    #widgets = ['Percentage of Granules: ', Percentage(), ' ', Bar(marker = RotatingMarker()), ' ', ETA(), ' ']
    #progress_bar = ProgressBar(widgets = widgets, maxval = granule_loader.number_of_granules).start()

    sums = numpy.zeros((lut_size, lut_size, lut_size))
    counts = numpy.zeros((lut_size, lut_size, lut_size))
    for index, granule in enumerate(granule_loader_chunks):
        try:
            new_lut = build_lookuptable({'data':granule[0].data, 'size':lut_size})
        except Exception as ex:
            continue

        '''
        prev_sum_counts = numpy.copy(all_avg_lut.counts)
        all_avg_lut.counts += new_lut.counts
        non_zero_locations = all_avg_lut.counts != 0

        all_avg_lut.table[non_zero_locations] = (new_lut.sums[non_zero_locations]/all_avg_lut.counts[non_zero_locations]) + \
                            ((prev_sum_counts[non_zero_locations] * all_avg_lut.table[non_zero_locations]) / all_avg_lut.counts[non_zero_locations])
        del new_lut
        del prev_sum_counts
        gc.collect()
        '''

        sums += new_lut.sums
        counts += new_lut.counts
        del new_lut
        gc.collect()
        #progress_bar.update(index)

    lut = sums/counts
    lut.tofile(str(lut_size) + '_lookuptable.numpy')
    counts.tofile(str(lut_size) + '_counts.numpy')
    sums.tofile(str(lut_size) + '_sums.numpy')




