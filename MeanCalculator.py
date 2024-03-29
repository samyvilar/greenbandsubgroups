__author__ = 'Samy Vilar'

import glasslab_cluster.cluster
import numpy
import scipy.spatial.distance
import scipy.cluster.vq
import networkx
import glasslab_cluster.cluster.consensus as gcons
import time
import pickle
import random
import gc
import sys

import Utils #load_cached_or_calculate_and_cached, multithreading_pool_map
from GranuleLoader import GranuleLoader
from labelling import labelling

import scipy.cluster.vq
import scipy.optimize

def append_ones(matrix, axis = 1):
    return numpy.append(matrix, numpy.ones((matrix.shape[0], 1)), axis = axis)


def minimize(**kwargs):
    initial_values  = kwargs['initial_values']
    function        = kwargs['function']
    max_iterations  = kwargs.get('max_iterations', None)

    return scipy.optimize.fmin(function, initial_values, maxiter = max_iterations, maxfun = max_iterations)[0]


def get_labels(data = None, means = None):
    assert data != None and means != None
    if len(means.shape) == 2 or len(means.shape) == 3:
        return labelling.get_labels(data = data, means = means)
    else:
        raise Exception("Can only handle regular clustering or sub-clustering ...")

def get_sub_values(data, labels, group):
    sub_labels = labels == group
    return data[sub_labels[:,0] & sub_labels[:,1], :]

def calc_predicted(kwargs):
    data                    = kwargs['data']
    training_band           = kwargs['training_band']
    alphas                  = kwargs['alphas']
    group                   = kwargs['group']
    labels                  = kwargs['labels']
    if len(alphas.shape) == 2:
        return numpy.dot(append_ones(data[:, training_band][labels == group]), alphas[group, :].reshape((-1,1)))
    elif len(alphas.shape) == 3:

        return numpy.asarray([
                    numpy.dot(append_ones(get_sub_values(data, labels, [group, subgroup])[:, training_band]),
                              alphas[group, subgroup, :].reshape((-1,1)))
                                    for subgroup in xrange(alphas.shape[1])])
    else:
        raise Exception("Only supporting clustering and sub-clustering!")

def is_empty_group(data = None, labels = None, group = None):
    pass

def check_for_empty_groups(data = None, labels = None, means = None):
    assert data != None and labels != None and means != None
    if len(labels.shape) == 1:
        while True:
            groups = xrange(means.shape[0])
            non_empty_groups = numpy.asarray([group in labels for group in groups], dtype = 'bool')
            if all(non_empty_groups):
                break
            empty_groups_indices = numpy.where(non_empty_groups == False)[0]
            new_groups = random.sample(data, len(empty_groups_indices))
            means[empty_groups_indices] = new_groups
            print "Empty group(s) %s new group(s) %s" % (str(empty_groups_indices), str(new_groups))
            labels = get_labels(data = data, means = means)
            sys.stdout.flush()
    elif len(labels.shape) == 2:
        groups = [[group, subgroup] for group in xrange(means.shape[0]) for subgroup in xrange(means.shape[1])]
        empty_groups = [any(( (labels[:, 0] == group[0]) & (labels[:, 1] == group[1]) )) for group in groups]
    else:
        raise Exception("Only supporting clustering and sub-clustering ...")

    return means, labels

    '''
    while not numpy.all(empty_groups):
        new_group = numpy.asarray([random.sample(data[:, index], 1)[0] for index in xrange(means.shape[1])])
        min_index = empty_groups.argmin()
        print "Empty label: %s group: %s new_group %s " % (str(min_index), str(means[min_index]), str(min_index))
        means[min_index] = new_group
        labels = get_labels(data = data, means = means)
        empty_groups = numpy.asarray([label in labels for label in xrange(means.shape[1])])
        sys.stdout.flush()
    '''

def get_predicted(**kwargs):
    data                    = kwargs['data']
    means                   = kwargs['means']
    training_band           = kwargs['training_band']
    predicting_band         = kwargs['predicting_band']
    enable_multithreading   = kwargs['enable_multithreading']

    labels  = get_labels(data = data, means = means)
    #check_for_empty_groups(data = data, labels = labels, means = means)
    values  = [dict(kwargs, group = group, labels = labels) for group in xrange(means.shape[0])]
    predictions = numpy.asarray(Utils.multithreading_pool_map(values = values, function = calc_predicted, multithreaded = enable_multithreading))

    #print 'predictions.shape %s' % str(predictions.shape)
    #print 'predictions %s' % str(predictions)
    predicted = numpy.zeros(data.shape[0])
    for index, value in enumerate(predictions):
        predicted[labels == index] = value
    pred = numpy.zeros(data.shape)
    pred[:, training_band] = data[:, training_band]
    pred[:, predicting_band[0] if len(predicting_band) == 1 else predicting_band] = predicted
    return pred

def calc_alpha(kwargs):
    data = kwargs['data']
    labels = kwargs['labels']
    group = kwargs['group']
    training_band = kwargs['training_band']
    predictive_band = kwargs['predictive_band']
    if len(labels.shape) == 1:
        c = data[labels == group, :]
        W = append_ones(c[:, training_band])
        G = c[:, predictive_band]
        return numpy.dot(numpy.linalg.inv(numpy.dot(W.T, W)), numpy.dot(W.T, G))
    elif len(labels.shape) == 2:
        means = kwargs['means']
        alphas = numpy.zeros(means.shape[1:])
        for subgroup in xrange(means.shape[1]):
            c = get_sub_values(data, labels, [group, subgroup])
            W = append_ones(c[:, training_band])
            G = c[:, predictive_band]
            alphas[subgroup, :] = numpy.column_stack(
                numpy.dot(numpy.linalg.inv(numpy.dot(W.T, W)), numpy.dot(W.T, G)))
        return alphas

def get_alphas(**kwargs):
    means = kwargs['means']
    enable_multithreading = kwargs['enable_multithreading']
    values = [dict(kwargs, group = group) for group in xrange(means.shape[0])]
    results = Utils.multithreading_pool_map(values = values,
                                    function = calc_alpha,
                                    multithreaded = enable_multithreading)
    if len(means.shape) == 2:
        return numpy.column_stack(results).transpose()
    elif len(means.shape) == 3:
        return numpy.asarray(results)
    else:
        raise Exception('Only supporting clustering and sub-clustering!')



def get_means(data, labels):
    assert data.ndim == 2 and labels.ndim == 1 and data.shape[0] == len(labels) and labels.min() >= 0
    number_of_clusters = labels.max() + 1
    means = numpy.zeros((number_of_clusters, data.shape[1]), dtype = 'f8')
    count = numpy.zeros(number_of_clusters, dtype = 'i')
    for i in xrange(number_of_clusters):
        indices = numpy.where(labels == i)[0]
        means[i,:] =  data[indices, :].mean(axis = 0)
        count[i] = len(indices)
    return means, count


def kmeans2_multithreading(kwargs):
    data = kwargs['data']
    number_of_groups = kwargs['number_of_groups']
    threshold = kwargs['threshold']
    number_of_runs = kwargs['number_of_runs']
    return scipy.cluster.vq.kmeans2(data, number_of_groups, thresh = threshold, iter = number_of_runs)

def get_mean(kwargs):
    data                                 = kwargs['data']
    number_of_runs                       = kwargs['number_of_runs']
    number_of_observations               = kwargs['number_of_observations']
    number_of_random_unique_sub_samples  = kwargs['number_of_random_unique_sub_samples']
    threshold                            = kwargs['threshold']

    number_of_points                     = kwargs['mean_shift'].number_of_points
    number_of_dimensions                 = kwargs['mean_shift'].number_of_dimensions
    number_of_neighbors                  = kwargs['mean_shift'].number_of_neighbors

    number_of_groups                     = kwargs['number_of_groups']
    number_of_sub_groups                 = kwargs['number_of_sub_groups']

    clustering_function                  = kwargs['clustering_function']


    assert numpy.all(numpy.isfinite(data))

    def clustering_function_kmeans2(data):
        means, labels = scipy.cluster.vq.kmeans2(data, number_of_groups, thresh = threshold, iter = number_of_runs)
        if number_of_sub_groups == 1:
            return means, labels

        labels = get_labels(data = data, means = means)
        values = [dict(data = data[labels == group],
                       number_of_groups = number_of_sub_groups,
                       threshold = threshold,
                       number_of_runs = number_of_runs) for group in xrange(means.shape[0])]
        results = Utils.multithreading_pool_map(values = values, function = kmeans2_multithreading, multithreaded = True)
        means = numpy.asarray([result[0] for result in results])
        sub_labels = numpy.asarray([result[1] for result in results])
        total_labels = numpy.zeros((labels.shape[0], 2))
        total_labels[:, 0] = labels
        for group in xrange(means.shape[0]):
            total_labels[labels == group, 1] = sub_labels[group]
        return means, total_labels


    def clustering_function_mean_shift(data):
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

    if clustering_function == 'mean_shift':
        run_labels, _ = gcons.subsampled(
            data,
            number_of_runs,
            clproc = pre_processing_function,
            cofunc = consensus_function,
            clfunc = clustering_function_mean_shift,
            nco    = number_of_observations,
            ncl    = number_of_random_unique_sub_samples)
        mrlabels = gcons.rmajrule(numpy.asarray(run_labels, dtype = 'int64'))

        means, count = get_means(data, mrlabels)
        return means

    if clustering_function == 'kmeans2':
        return clustering_function_kmeans2(data)


    raise Exception("Need to specify a clustering function!")


def get_predicted_from_means(**kwargs):
    data = kwargs['data']
    means = kwargs['means']
    original = kwargs['original']
    training_band = kwargs['training_band']
    predictive_band = kwargs['predictive_band']
    enable_multithreading = kwargs['enable_multithreading']

    labels = get_labels(data = data, means = means)
    sys.stdout.flush()
    means, labels = check_for_empty_groups(data = data, labels = labels, means = means)

    alphas = get_alphas(data = data,
                        means = means,
                        labels = labels,
                        training_band = training_band,
                        predictive_band = predictive_band,
                        enable_multithreading = enable_multithreading)

    gc.collect()
    return get_predicted(data = original,
                        means = means,
                        alphas = alphas,
                        training_band = training_band,
                        predicting_band = predictive_band,
                        enable_multithreading = enable_multithreading)

def calc_means(**kwargs):
    def getmeans(data = None, labels = None):
        number_clusters = labels.max() + 1
        mu = numpy.zeros((number_clusters, data.shape[1]))
        for i in xrange(number_clusters):
            mu[i,:] = data[labels == i].mean(axis = 0)
        return mu
    files_and_clustering_properties  = kwargs['files_clustering_properties']
    number_sub_groups                = kwargs['number_of_sub_groups']

    if len(files_and_clustering_properties) == 0:
        return []

    results   = Utils.multithreading_pool_map(values       = files_and_clustering_properties, # calculate means for each granule
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
        return getMeans(kwargs['hdf_file'].data, labels = kmeans2(kwargs['hdf_file'].data, kwargs['number_of_sub_groups'], threshold = kwargs['threshold'])[1])

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
        self._number_of_sub_groups                   = None
        self._number_of_runs                        = None
        self._number_of_random_unique_sub_samples   = None
        self._number_of_observations                = None
        self._threshold                             = None
        self._labels                                = None
        self._mean_shift                            = MeanShift()
        self._means                                 = None

        self.enable_multithreading()

        self._required_properties = ['number_of_groups',
                                     'number_of_sub_groups',
                                     'number_of_runs',
                                     'threshold',
                                     'number_of_observations',
                                     'number_of_random_unique_sub_samples',
                                     'clustering_function',
                                     'mean_shift']

    @property
    def clustering_function(self):
        return self._clustering_function
    @clustering_function.setter
    def clustering_function(self, value):
        self._clustering_function = value

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
    def labels(self):
        return self._labels
    @labels.setter
    def labels(self, values):
        self._labels = values

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
            values[-1].update(self.get_properties_as_dict())

        return values

    def get_properties_as_dict(self):
        props = {}
        for property in self._required_properties:
            props[property] = getattr(self, property)
        return props


    def calc_caching_file_name(self):
        if self.granules == None or len(self.granules) == 0: return "None"
        return '%s/number_of_granules:%i_param:%s_bands:%s_names_hashed:%s_number_of_groups:%s_number_of_sub_groups:%i_initial_means.obj' % \
            (self.granules[0].file_dir + '/cache/means', len(self.granules), self.granules[0].param, str(self.granules[0].bands), GranuleLoader.get_names_hashed([granule.file_name for granule in self.granules]), self.number_of_groups, self.number_of_sub_groups)

    def calculate_means_data(self, data, function = get_mean):
        props = self.get_properties_as_dict()
        props['data'] = data

        self.means, self.labels = get_mean(props)
        return self.means, self.labels

    def predict(self, original):
        assert self.means



    def calculate_means(self):
        self.check_all_properties()
        self.means = Utils.load_cached_or_calculate_and_cached(
            caching     = self.is_caching(),
            file_name   = self.calc_caching_file_name(),
            function    = calc_means,
            arguments   =
                {
                    'files_clustering_properties':self.get_clustering_properties_as_array_dict_for_each_file(),
                    'clustering_function':get_mean,
                    'multithreaded':self.is_multithreading(),
                })
