__author__ = 'Samy Vilar'

import glasslab_cluster.cluster
import numpy
from os.path import basename
import scipy.spatial.distance
import scipy.cluster.vq
import networkx
import glasslab_cluster.cluster.consensus as gcons
import time
import pickle
import scipy.cluster.vq.kmeans2

from Utils import load_cached_or_calculate_and_cached, multithreading_pool_map
from GranuleLoader import GranuleLoader


def getMeans(data, labels):
    assert data.ndim == 2 and labels.ndim == 1 and data.shape[0] == len(labels) and labels.min() >= 0
    number_of_clusters = labels.max() + 1
    means = numpy.zeros((number_of_clusters, data.shape[1]), dtype = 'f8')
    count = numpy.zeros(number_of_clusters, dtype = 'i')
    for i in xrange(number_of_clusters):
        indices = numpy.where(labels == i)[0]
        means[i,:] =  data[indices, :].mean(axis = 0)
        count[i] = len(indices)
    return means, count


def getMean(kwargs):

    hdf_file                             = kwargs['hdf_file']
    number_of_runs                       = kwargs['number_of_runs']
    number_of_observations               = kwargs['number_of_observations']
    number_of_random_unique_sub_samples  = kwargs['number_of_random_unique_sub_samples']
    threshold                            = kwargs['threshold']

    number_of_points                     = kwargs['mean_shift'].number_of_points
    number_of_dimensions                 = kwargs['mean_shift'].number_of_dimensions
    number_of_neighbors                  = kwargs['mean_shift'].number_of_neighbors


    data = hdf_file.data

    if numpy.any(data.min(axis = 0) == data.max(axis = 0)):
        print "Skipping %s" % basename(hdf_file.get_file()) # competitive learning will crash, so lets skip it
        return None

    assert numpy.all(numpy.isfinite(data))

    def clustering_function(data):
        def mean_shift(data):
            K = number_of_points  # n is the number of points
            L = number_of_dimensions   # d is the number of dimensions.
            k = number_of_neighbors # number of neighbors
            f = glasslab_cluster.cluster.FAMS(data, seed = 100) #FAMS Fast Adaptive Mean Shift

            pilot = f.RunFAMS(K, L, k)
            modes = f.GetModes()
            umodes = glasslab_cluster.utils.uniquerows(modes)
            labels = numpy.zeros(modes.shape[0])
            for i, m in enumerate(umodes):
                labels[numpy.all(modes == m, axis = 1)] = i
            return umodes, labels, pilot


        means, sub_labels, pilot = mean_shift(data)
        print 'means.shape' + str(means.shape)
        distance_matrix = scipy.spatial.distance.pdist(means)
        print "distance matrix min max:", distance_matrix.min(), distance_matrix.max()
        distance_matrix[distance_matrix > threshold] = 0
        H = networkx.from_numpy_matrix(scipy.spatial.distance.squareform(distance_matrix))
        connected_components = networkx.connected_components(H)

        print len(connected_components), "components:", map(len, connected_components)

        def merge_cluster(pattern, lbl_composites):
            try:
                pattern.shape #test if pattern is a NUMPY array, convert if list
            except:
                pattern = numpy.array(pattern)
            for i, composite in enumerate(lbl_composites):
                for label in composite:
                    if label != i:
                        pattern[numpy.where(pattern == label)] = i
            return pattern

        labels = merge_cluster(sub_labels, connected_components) # modify in order  to merge means ...
        return labels

    def consensus_function(run_labels):
        return gcons.BestOfK(run_labels)

    def pre_processing_function(data):
        time.sleep(1)
        return scipy.cluster.vq.whiten(data - data.mean(axis = 0))

    run_labels, _ = gcons.subsampled(
            data,
            number_of_runs,
            clproc = pre_processing_function,
            cofunc = consensus_function,
            clfunc = clustering_function,
            nco    = number_of_observations,
            ncl    = number_of_random_unique_sub_samples)


    mrlabels = gcons.rmajrule(numpy.asarray(run_labels, dtype = 'int64'))

    means, count = getMeans(data, mrlabels)
    print means
    return means

def calc_means(**kwargs):
    def getmeans(data = None, labels = None):
        number_clusters = labels.max() + 1
        mu = numpy.zeros((number_clusters, data.shape[1]))
        for i in xrange(number_clusters):
            mu[i,:] = data[labels == i].mean(axis = 0)
        return mu
    files_and_clustering_properties  = kwargs['files_clustering_properties']
    number_sub_groups                = kwargs['number_of_sub_groups']

    if len(files_and_clustering_properties) == 0: return []

    results   = multithreading_pool_map(values       = files_and_clustering_properties, # calculate means for each granule
                                       function      = kwargs['clustering_function'],
                                       multithreaded = kwargs['multithreaded'])

    print results
    pickle.dump(results, open('results.obj', 'wb'))
    results = numpy.asarray(results)

    means = getmeans(data = results,
                    labels = glasslab_cluster.cluster.aghc(
                        results,
                        files_and_clustering_properties[0]['number_of_groups'],
                        method = 'max',
                        metric = 'cityblock'))

    print means
    pickle.dump(means, open('means.mat', 'wb'))

    def calc_means_sub_group(**kwargs):
        return getMeans(kwargs['hdf_file'].data, labels = scipy.cluster.vq.kmeans2(kwargs['hdf_file'].data, kwargs['number_of_sub_groups'], threshold = kwargs['threshold'])[1])




    return means





class MeanShift(object):
    def __init__(self, number_of_points = None, number_of_dimensions = None, number_of_neighbors = None):
        self._number_of_points      = number_of_points
        self._number_of_dimensions  = number_of_dimensions
        self._number_of_neighbors   = number_of_neighbors

    @property
    def number_of_points(self):
        return self._number_of_points
    @property
    def number_of_dimensions(self):
        return self._number_of_dimensions
    @property
    def number_of_neighbors(self):
        return self._number_of_neighbors


class MeanCalculator(object):
    def __init__(self):
        self._granules                              = None
        self._number_of_groups                      = None
        self._number_of_subgroups                   = None
        self._number_of_runs                        = None
        self._number_of_random_unique_sub_samples   = None
        self._number_of_observations                = None
        self._threshold                             = None
        self._labels                                = None
        self._mean_shift                            = None
        self._means                                 = None

        self.enable_multithreading()

        self._required_properties = ['number_of_groups',
                                     'number_of_subgroups',
                                     'number_of_runs',
                                     'number_of_random_unique_sub_samples',
                                     'number_of_observations',
                                     'threshold',
                                     'mean_shift']

    @property
    def granules(self):
        return self._granules
    @granules.setter
    def granules(self, values):
        self._granules = values

    @property
    def number_of_groups(self):
        return self._number_of_groups
    @number_of_groups.setter
    def number_of_groups(self, values):
        self._number_of_groups = values

    @property
    def number_of_sub_groups(self):
        return self._number_of_sub_groups
    @number_of_sub_groups.setter
    def number_of_sub_groups(self, values):
        self._number_of_sub_groups = values

    @property
    def number_of_runs(self):
        return self._number_of_runs
    @number_of_runs.setter
    def number_of_runs(self, values):
        self._number_of_runs = values

    @property
    def number_of_random_unique_sub_samples(self):
        return self._number_of_random_unique_sub_samples
    @number_of_random_unique_sub_samples.setter
    def number_of_random_unique_sub_samples(self, values):
        self._number_of_random_unique_sub_samples = values

    @property
    def number_of_observations(self):
        return self._number_of_observations
    @number_of_observations.setter
    def number_of_observations(self, values):
        self._number_of_observations = values

    @property
    def threshold(self):
        return self._threshold
    @threshold.setter
    def threshold(self, values):
        self._threshold = values

    @property
    def labels(self):
        return self._labels
    @labels.setter
    def labels(self, values):
        self._labels = values

    @property
    def mean_shift(self):
        return self._mean_shift
    @mean_shift.setter
    def mean_shift(self, values):
        self._mean_shift = values

    @property
    def means(self):
        return self._means
    @means.setter
    def means(self, values):
        self._means = values

    @property
    def granule_loader(self):
        return self._granule_loader
    @granule_loader.setter
    def granule_loader(self, values):
        self._granule_loader = values
        if self.granule_loader.state == "LOADED":
            self.granules = self.granule_loader.granules
        else:
            raise Exception("Granule Loader must be set and load the granules!")


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


    def check_all_properties(self):
        for property in self._required_properties:
            if getattr(self, property) == None:
                raise Exception("You must set the following property %s" % property)

    def get_clustering_properties_as_array_dict_for_each_file(self):
        values = []
        for file in self.granules:
            values.append({})
            values[-1]['hdf_file'] = file
            for property in self._required_properties:
                values[-1][property] = getattr(self, property)

        return values

    def calc_caching_file_name(self):
        if self.granules == None or len(self.granules) == 0: return "None"
        return '%s/number_of_granules:%i_param:%s_bands:%s_names_hashed:%s_number_of_groups:%s_number_of_subgroups:%i_initial_means.obj' % \
            (self.granules[0].file_dir + '/cache/means', len(self.granules), self.granules[0].param, str(self.granules[0].bands), GranuleLoader.get_names_hashed([granule.file_name for granule in self.granules]), self.number_of_groups, self.number_of_subgroups)


    def calculate_means(self):
        self.check_all_properties()
        self.means = load_cached_or_calculate_and_cached(
            caching     = self.is_caching(),
            file_name   = self.calc_caching_file_name(),
            function    = calc_means,
            arguments   =
                {
                    'files_clustering_properties':self.get_clustering_properties_as_array_dict_for_each_file(),
                    'clustering_function':getMean,
                    'multithreaded':self.is_multithreading(),
                })






