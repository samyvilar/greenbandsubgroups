
import sys
sys.path.append('../../..')

from GranuleLoader import GranuleLoader
import numpy


granule_loader = GranuleLoader()
granule_loader.bands = numpy.asarray([1,2,3,4])
granule_loader.param = 'reflectance'
granule_loader.disable_caching()
granule_loader.enable_multithreading()
granule_loader.load_granules_from_dir(dir = '/home1/FoucaultData/DATA_11/TERRA_1KM', pattern = '*.hdf')

