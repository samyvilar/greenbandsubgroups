__author__ = 'Samy Vilar'
__date__ = "Jan 5, 2012"

import os, os.path
import os
import pprint
import glasslab_cluster.io
import numpy
import multiprocessing
import pickle

from HDFFile import HDFFile
from Utils import load_cached_or_calculate_and_cached, multithreading_pool_map
from MeanCalculator import MeanCalculator


class GreenBandSubGroup(object):
    log = []
    def __init__(self):
        self.log_level = 10000
        self.enable_multithreading()

    def enable_multithreading(self):
        self._multithreaded = True
    def disable_multithreading(self):
        self._multithreaded = False
    def is_multithreading(self):
        return self._multithreaded

    @staticmethod
    def enter_log(log):
        GreenBandSubGroup.log.append(log)

    def set_debug_level(self, level):
        self.log_level = level

    def add_granule(self, granule):
        self.granules.append(granule)













