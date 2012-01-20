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
        hash = 0
        for name in names:
            for ch in name:
                hash += ord(ch)
        return hash % 10**8


    def calc_granules_cached_file_name(self, granules = None):
        assert granules
        return 'number_of_granules:%i_param:%s_bands:%s_crop_size:%s_crop_ori:%s_names_hashed:%s' %\
               (len(granules), str(self.param), str(self.bands).strip(' '), str(self.crop_size), str(self.crop_orig), GranuleLoader.get_names_hashed([granule.file_name for granule in granules]))



    def load_granules(self, granules):
        self._verify_properties()
        for index, granule in enumerate(granules):
            if not isinstance(granule, HDFFile):
                granules[index] = HDFFile(granule)
            granules[index].bands       = self.bands
            granules[index].set_param   = self.param
            granules[index].crop_size   = self.crop_size
            granules[index].crop_orig   = self.crop_orig

        self.granules = load_cached_or_calculate_and_cached(
                            caching = self.is_caching(),
                            file_name = self.calc_granules_cached_file_name(granules = granules) if self.is_caching() else None,
                            function = multithreading_pool_map,
                            arguments =
                            {
                                'values':granules,
                                'function':load_granules_threaded,
                                'multithreaded':self.is_multithreading(),
                            }
                        )
        

    def load_granules_from_dir(self, dir = None, pattern = None):
        assert dir and pattern
        self._verify_properties()
        self.dir = dir
        files = [dir + '/' + file for file in os.listdir(dir) if fnmatch.fnmatch(file, pattern)]
        self.load_granules(files)
