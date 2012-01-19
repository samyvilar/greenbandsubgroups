__author__ = 'Samy Vilar'

import fnmatch, os
from HDFFile import HDFFile
from Utils import multithreading_pool_map, load_cached_or_calculate_and_cached

class GranuleLoader(object):
    def __init__(self):
        self._bands     = None
        self._param     = None
        self._crop_size = None
        self._crop_orig = None
        self._dir       = None
        self._granules  = None

    @property
    def param(self):
        return self._param
    @param.setter
    def param(self, value):
        self._param = value
    @property
    def bands(self):
        return self._bands
    @bands.setter
    def bands(self, value):
        self._bands = value
    @property
    def crop_size(self):
        return self._crop_size
    @crop_size.setter
    def crop_size(self, value):
        self._crop_size = value
    @property
    def crop_orig(self):
        return self._crop_orig
    @crop_orig.setter
    def crop_orig(self, value):
        self._crop_orig = value

    @property
    def dir(self):
        return self._dir
    @dir.setter
    def dir(self, value):
        self._dir = value

    @property
    def granules(self):
        return self._granules
    @granules.setter
    def granules(self, values):
        self._granules = values

    def enable_caching(self):
        self._caching = True
    def disable_caching(self):
        self._caching = False
    def is_caching(self):
        return self._caching

    def enable_multithreading(self):
        self._multithreading = True
    def disable_multithreading(self):
        self._multithreading = False
    def is_multithreading(self):
        return self._multithreading


    def _verify_properties(self):
        assert self.param != None and self.bands != None

    @staticmethod
    def get_names_hashed(names):
        pass

    def calc_granules_cached_file_name(self):
        return 'number_of_granules:%i_param:%s_bands:%s_crop_size:%s_crop_ori:%s_names_hashed:%s' %\
               (len(self.get_granules()), str(self.param), str(self.bands), str(self.crop_size), str(self.crop_orig), GranuleLoader.get_names_hashed(self.granules))



    def load_granules(self, granules):
        self._verify_properties()
        for index, granule in enumerate(granules):
            if not isinstance(granule, HDFFile):
                granules[index] = HDFFile(granule)
            granule.bands       = self.bands
            granule.set_param   = self.param
            granule.crop_size   = self.crop_size
            granule.crop_orig   = self.crop_orig

        caching_file = self.calc_granules_cached_file_name(granules = granules)
        if self.is_caching():
            caching_func = load_cached_or_calculate_and_cached
        else:
            caching_func = lambda file, func, *args: func(args)

        if self.is_multithreading():
            def load_granules_threaded(granule):
                try:
                    granule.load()
                except Exception as ex:
                    print str(ex)
                    return None
                return granule
            self.granules = caching_func(caching_file, multithreading_pool_map, (self.get_granules(), load_granules_threaded))
        else:
            self.granules =  caching_func(caching_file,
                lambda granules: [granule for granule in (lambda granules: [granule.load() for granule in granules])(granules) if granule], self.granules)


    def load_granules_from_dir(self, dir = None, pattern = None):
        assert dir and pattern
        self._verify_properties()
        self.dir = dir
        files = [dir + '/' + file for file in os.listdir(dir) if fnmatch.fnmatch(file, pattern)]
        self.load_granules(files)
