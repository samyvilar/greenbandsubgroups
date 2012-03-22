__author__ = 'Samy Vilar'

import sys
sys.path.extend('../../../../../..')

import numpy

from lookuptable.lookuptable import lookuptable
from GranuleLoader import GranuleLoader
from matplotlib import pyplot as plt

if __name__ == '__main__':
    lut = lookuptable()
    lut.load_table('../800_lookuptable.numpy')

    granule_loader = GranuleLoader()
    granule_loader.bands = [1,2,3,4]
    granule_loader.param = 'reflectance'
    granule_loader.disable_caching()
    granule_loader.enable_multithreading()

    granule_loader.load_granules(granules = ['/home1/FoucaultData/DATA_11/TERRA_1KM/MOD021KM.A2010172.1200.005.2010172204449.hdf',])
    original = granule_loader.granules[0].data
    predicted = lut.predict(original)

    std = numpy.sqrt(numpy.sum((predicted[:, 3] - original[:, 3])**2)/original[:, 3].shape[0])
    print std/numpy.mean(original[:,3]) * 100

    original_shape = granule_loader.granules[0].original_shape[0:2]
    original = original[:, 3].reshape(original_shape)
    predicted = predicted[:, 3].reshape(original_shape)

    plt.figure()
    plt.imshow(original)
    plt.colorbar()
    plt.savefig('original.png')

    plt.figure()
    plt.imshow(predicted)
    plt.colorbar()
    plt.savefig('predicted.png')


    plt.figure()
    plt.imshow(numpy.fabs(original - predicted))
    plt.colorbar()
    plt.savefig('error.png')



