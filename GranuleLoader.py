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
        self.__state    = "INITIAL"

    @property
    def state(self):
        return self.__state

    @property
    def _state(self):
        return self.__state
    @_state.setter
    def _state(self, values):
        self.__state = values

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

    def enable_caching(self, name = None):
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


    @staticmethod
    def calc_granules_cached_file_name(granules = None):
        if granules == None or len(granules) == 0: return "None"
        return '%s/number_of_granules:%i_param:%s_bands:%s_crop_size:%s_crop_ori:%s_names_hashed:%s' %\
               (granules[0].file_dir + '/cache/granules', len(granules), str(granules[0].param), str(granules[0].bands).strip(' '), str(granules[0].crop_size), str(granules[0].crop_orig), GranuleLoader.get_names_hashed([granule.file_name for granule in granules]))


    @property
    def number_of_granules(self):
        return self._number_of_granules
    @number_of_granules.setter
    def number_of_granules(self, values):
        self._number_of_granules = values

    def load_granules(self, granules = []):
        self._verify_properties()
        for index, granule in enumerate(granules):
            if not isinstance(granule, HDFFile):
                granules[index] = HDFFile(granule)
            granules[index].bands       = self.bands
            granules[index].param       = self.param
            granules[index].crop_size   = self.crop_size
            granules[index].crop_orig   = self.crop_orig

        granules = load_cached_or_calculate_and_cached(
                            caching = self.is_caching(),
                            file_name = self.calc_granules_cached_file_name(granules = granules),
                            function = multithreading_pool_map,
                            arguments =
                            {
                                'values':[{'granule':granule} for granule in granules],
                                'function':load_granules_threaded,
                                'multithreaded':self.is_multithreading(),
                            }
                        )
        self.granules = [granule for granule in granules if granule]
        self._state = "LOADED"


    @staticmethod
    def get_granules_in_dir(dir = None, pattern = None):
        return [dir + '/' + file for file in os.listdir(dir) if fnmatch.fnmatch(file, pattern)]


    def load_granules_from_dir(self, dir = None, pattern = None):
        assert dir and pattern
        self._verify_properties()
        self.dir = dir
        files = GranuleLoader.get_granules_in_dir(dir = dir, pattern = pattern)
        self.load_granules(granules = files)

    def load_granules_chunk(self, dir = None, pattern = None, chunks = None, max = None):
        assert chunks and dir and pattern
        files = GranuleLoader.get_granules_in_dir(dir = dir, pattern = pattern)
        index = 0
        self.number_of_granules = max if max else len(files)
        while index < self.number_of_granules:
            self.load_granules(granules = files[index:index+chunks])
            index += chunks
            yield self.granules




def load_granules_threaded(kwvalues):
    granule = kwvalues['granule']
    try:
        granule.load()
    except Exception as ex:
        print "Failed to load granule %s Error: %s" % (str(granule.file), str(ex))
        return None
    return granule