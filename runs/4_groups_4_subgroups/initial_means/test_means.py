__author__ = 'Samy Vilar'
__date__ = 'Mar 22, 2012'

import sys
sys.path.extend('../../..')

from lookuptable.lookuptable import  lookuptable
from MeanCalculator import MeanCalculator, MeanShift

lut = lookuptable()
lut.load_table('../../lookuptable/reflectance/800/800_lookuptable.numpy')
lut.load_or_calulate_flatten_table('../../lookuptable/reflectance/800/800_lookuptable_flatten.numpy')


mean_calculator = MeanCalculator()                  # Calculate or load initial means ...
mean_calculator.enable_multithreading()
mean_calculator.disable_caching()

mean_calculator.threshold = .8
mean_calculator.number_of_groups = 4
mean_calculator.number_of_sub_groups = 1
mean_calculator.number_of_runs = 10
mean_calculator.number_of_random_unique_sub_samples  = 1000
mean_calculator.number_of_observations = 274862
mean_calculator.mean_shift = MeanShift(number_of_points = 30, number_of_dimensions = 1, number_of_neighbors = 100)

mean_calculator.calculate_means()

mean = get_mean(**{'data':lut.table,
                   'number_of_runs':''})