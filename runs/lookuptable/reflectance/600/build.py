import gc
gc.enable()

import sys
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

chunk = granule_loader.load_granules_chunk(dir = '/DATA_11/TERRA_1KM', pattern = '*.hdf', chunks = 1).next()
t = build_lookuptable({'data':chunk[0].data, 'size':1000})

tables = multithreading_pool_map(function = build_lookuptable, values = [{'data':c.data, 'size':600} for c in chunk], multithreaded = True)



