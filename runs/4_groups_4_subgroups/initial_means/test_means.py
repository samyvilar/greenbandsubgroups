__author__ = 'Samy Vilar'
__date__ = 'Mar 22, 2012'

import sys
sys.path.extend('../../..')

from lookuptable.lookuptable import  lookuptable
from MeanCalculator import MeanCalculator, MeanShift
lut = lookuptable()
lut.load_flatten_table('../../lookuptable/reflectance/800/800_lookuptable_flatten.numpy')
data = lut.indices_to_data(lut.flatten_table)


mean_calculator = MeanCalculator()                  # Calculate or load initial means ...
mean_calculator.enable_multithreading()
mean_calculator.disable_caching()

mean_calculator.threshold = .15
mean_calculator.number_of_groups = 8
mean_calculator.number_of_sub_groups = 1
mean_calculator.number_of_runs = 10
mean_calculator.number_of_random_unique_sub_samples  = 3000 #data.shape[0]*data.shape[1] * .001
mean_calculator.number_of_observations = data.shape[0]*data.shape[1]
mean_calculator.mean_shift = MeanShift(number_of_points = 30, number_of_dimensions = 1, number_of_neighbors = 100)


mean = mean_calculator.calculate_mean(data = data)
