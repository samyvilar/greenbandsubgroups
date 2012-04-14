__author__ = 'Samy Vilar'
__date__ = 'Mar 22, 2012'

import sys
sys.path.extend('../../..')

from lookuptable.lookuptable import  lookuptable
from MeanCalculator import MeanCalculator, MeanShift
from GranuleLoader import GranuleLoader
import numpy

lut = lookuptable()
lut.load_flatten_table('../../lookuptable/reflectance/800/800_lookuptable_flatten.numpy')
data = lut.indices_to_data(lut.flatten_table)



granule_loader = GranuleLoader()
granule_loader.bands = [1,2,3,4]
granule_loader.param = 'reflectance'
granule_loader.disable_caching()
granule_loader.enable_multithreading()

granule_path = '/DATA_5/SNOW_CLOUD_MODIS/data/MOD021KM.A2002179.1640.005.2010085164818.hdf'
granule_loader.load_granules(granules = [granule_path,])
original = granule_loader.granules[0].data


mean_calculator = MeanCalculator()                  # Calculate or load initial means ...
mean_calculator.enable_multithreading()
mean_calculator.enable_caching()

mean_calculator.threshold = 1e-05
mean_calculator.number_of_groups = 8
mean_calculator.number_of_sub_groups = 1
mean_calculator.number_of_runs = 10
mean_calculator.number_of_random_unique_sub_samples  = 3000 #data.shape[0]*data.shape[1] * .001
mean_calculator.number_of_observations = data.shape[0]*data.shape[1]
mean_calculator.mean_shift = MeanShift(number_of_points = 30, number_of_dimensions = 1, number_of_neighbors = 100)
mean_calculator.clustering_function = "kmeans2"

means, labels = mean_calculator.calculate_means_data(data)


alphas = numpy.dot(
    numpy.dot(numpy.linalg.inv(numpy.dot(data[:, 0:3].transpose(), data[:, 0:3])),
        data[:, 0:3].transpose()), data[:, 3])





