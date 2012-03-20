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
    lut_size = 600

    all_avg_lut = build_lookuptable({'data':granule_loader_chunks.next()[0].data, 'size':lut_size})
    gc.collect()

    widgets = ['Percentage of Granules: ', Percentage(), ' ', Bar(marker = RotatingMarker()), ' ', ETA(), ' ']
    progress_bar = ProgressBar(widgets = widgets, maxval = granule_loader.number_of_granules).start()

    for index, granule in enumerate(granule_loader_chunks):
        new_lut = build_lookuptable({'data':granule[0].data, 'size':lut_size})

        prev_sum_counts = numpy.copy(all_avg_lut.counts)
        all_avg_lut.counts += new_lut.counts
        non_zero_locations = all_avg_lut.counts != 0

        temp_1 = (new_lut.sums[non_zero_locations]/all_avg_lut.counts[non_zero_locations])

        print 'non_zero_locations->' + str(non_zero_locations)
        print 'non_zero_locations.shape->' + str(non_zero_locations.shape)
        print 'all_avg_lut.table->' + str(all_avg_lut.table)
        print 'all_avg_lut.table.shape->' + str(all_avg_lut.table.shape)
        print all_avg_lut.table[non_zero_locations]
        print prev_sum_counts[non_zero_locations]


        temp_2 = ((prev_sum_counts[non_zero_locations] * all_avg_lut.table[non_zero_locations]))
        temp_2 /= all_avg_lut.counts[non_zero_locations]

        all_avg_lut.table = temp_1 + temp_2

        del new_lut
        del prev_sum_counts
        gc.collect()

        progress_bar.update(index)

    all_avg_lut.table.tofile(str(lut_size) + '_lookuptable.numpy')
    all_avg_lut.counts.tofile(str(lut_size) + '_size.numpy')




