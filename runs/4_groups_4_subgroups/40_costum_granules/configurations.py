__author__ = 'Samy Vilar'

import numpy

from GranuleLoader import GranuleLoader
from MeanCalculator import MeanCalculator, MeanShift


granules = [
    '/DATA_6/TERRA_1KM/MOD021KM.A2010170.0045.005.2010170084123.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010170.0050.005.2010170084357.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010170.0110.005.2010170083549.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.0055.005.2010172085036.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.0100.005.2010172085124.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.0235.005.2010172094806.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.0710.005.2010172175208.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.0720.005.2010172175152.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.0725.005.2010172174733.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.0730.005.2010172174941.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.0900.005.2010172194449.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.0905.005.2010172194613.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.0910.005.2010172195010.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.0915.005.2010172194605.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.1040.005.2010172195057.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.1405.005.2010172221649.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.1410.005.2010172221716.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.1900.005.2010173022237.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.2020.005.2010173042549.hdf',

    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.1345.005.2010172210518.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.1520.005.2010173012743.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.1700.005.2010173012451.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010170.0220.005.2010170104559.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.1340.005.2010172205256.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.0350.005.2010172163735.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010173.0115.005.2010173130627.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.0030.005.2010172085226.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.1840.005.2010173021315.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010170.0050.005.2010170084357.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.0525.005.2010172164119.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010170.0040.005.2010170083932.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.0210.005.2010172094915.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.1205.005.2010172205155.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.1525.005.2010173012722.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.2015.005.2010173042349.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.0345.005.2010172163830.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.0705.005.2010172174416.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.2335.005.2010173072211.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.2155.005.2010173072138.hdf',
    '/DATA_6/TERRA_1KM/MOD021KM.A2010172.1025.005.2010172194922.hdf'
]



if __name__ == '__main__':
    granule_loader = GranuleLoader()                    # Load Granules
    granule_loader.bands = numpy.asarray([1,2,3,4])
    granule_loader.param = 'radiance'
    granule_loader.disable_caching()
    granule_loader.enable_multithreading()
    granule_loader.load_granules(granules)


    mean_calculator = MeanCalculator()                  # Calculate or load initial means ...
    mean_calculator.enable_multithreading()
    mean_calculator.enable_caching()

    mean_calculator.granule_loader = granule_loader
    mean_calculator.threshold = .8
    mean_calculator.number_of_groups = 8
    mean_calculator.number_of_subgroups = 1
    mean_calculator.number_of_runs = 10
    mean_calculator.number_of_random_unique_sub_samples  = 1000
    mean_calculator.number_of_observations = 274862
    mean_calculator.mean_shift = MeanShift(number_of_points = 30, number_of_dimensions = 1, number_of_neighbors = 100)

    mean_calculator.calculate_means()







