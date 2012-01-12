__author__ = 'Samy Vilar'
__date__ = "Jan 5, 2012"

import os, os.path
import fnmatch
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

    def set_bands(self, bands):
        self.bands = bands
        GreenBandSubGroup.enter_log("Set %s bands." % str(self.bands))

    def set_granules(self, granules):
        self._granules = granules
        GreenBandSubGroup.enter_log("Set %i granules: \n %s" % (len(self.granules), pprint.pformat(self.granules)))

    def get_granules(self):
        if hasattr(self, 'granules'):
            return self._granules
        else:
            raise Exception("The granules haven't being set!")

    def set_granules_dir(self, dir, pattern):
        if not os.path.isdir(dir):
            raise Exception("Dir %s is not a directory!" % dir)
        self.granules = [HDFFile(dir + '/' + file) for file in os.listdir(dir) if fnmatch.fnmatch(file, pattern)]
        self.set_granules(self.granules)

    def add_granule(self, granule):
        self.granules.append(granule)

    def load_granules(self, dir = None, pattern = None, bands = None, param = None, crop_size = None, crop_orig = None, caching = None):
        assert dir and pattern and bands and param
        self.set_granules_dir(dir, pattern)
        for granule in self.get_granules():
            granule.set_bands(bands)
            granule.set_param(param)
            granule.set_crop_size(crop_size)
            granule.set_crop_orig(crop_orig)
        prev_granules_len = len(self.get_granules())

        cached_granule_file = dir + "/granules.bin"
        if caching == True:
            caching_func = load_cached_or_calculate_and_cached
        else:
            caching_func = lambda file, func, *args: func(args)


        if self.is_multithreading():
            def load_granules_threaded(granule):
                try:
                    granule.load()
                except Exception as ex:
                    return None
                return granule
            self.set_granules(caching_func(cached_granule_file, multithreading_pool_map, (self.get_granules(), load_granules_threaded)))
        else:
            self.set_granules(caching_func(
                cached_granule_file,
                lambda granules:
                    [granule for granule in (lambda granules: [granule.load() for granule in granules])(granules) if granule],
                self.get_granules()))

        GreenBandSubGroup.enter_log("Loaded %i out of %i granules." % (len(self.get_granules()), prev_granules_len))


    def test_func(self, func, *args):
        self.__getattribute__(func)(args)













