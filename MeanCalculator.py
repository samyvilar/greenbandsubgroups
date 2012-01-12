__author__ = 'Samy Vilar'

import glasslab_cluster.cluster
import numpy
from os.path import basename
import scipy.spatial.distance
import networkx
import glasslab_cluster.cluster.consensus as gcons
import time

from Utils import load_cached_or_calculate_and_cached, multithreading_pool_map


def getMean(kwargs):

    hdf_file                            = kwargs['hdf_file']
    number_of_runs                      = kwargs['number_of_runs']
    number_of_observations              = kwargs['number_of_observations']
    number_of_random_unique_subsamples  = kwargs['number_of_random_unique_subsamples']
    threshold                           = kwargs['threshold']

    number_of_points                    = kwargs['number_of_points']
    number_of_dimensions                = kwargs['number_of_dimensions']
    number_of_neighbors                 = kwargs['number_of_neighbors']


    data = hdf_file.get_data()

    if numpy.any(data.min(axis = 0) == data.max(axis = 0)):
        print "Skipping %s" % basename(hdf_file.get_file()) # competive learning will crash, so lets skip it
        return None

    assert numpy.all(numpy.isfinite(data))

    def clustering_function(data):
        def mean_shift(data):
            K = number_of_points  #  n is the number of points
            L = number_of_dimensions   #  d is the number of dimensions.
            k = number_of_neighbors #  number of neighbors
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
            ncl    = number_of_random_unique_subsamples)
    mrlabels = gcons.rmajrule(run_labels)

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

    means, count = getMeans(data, mrlabels)
    return means


def calc_means(values, threaded):
    def save_reducedmeans(means, meansfile):
        means = numpy.row_stack(means)
        print "mean:", means.shape, means.min(), means.max()
        newmeans = self.getmeans(means, glasslab_cluster.cluster.aghc(means, NNewMeans, method='max', metric='cityblock'))
        print "newmean:", newmeans.shape, newmeans.min(), newmeans.max()
        plt.figure()
        for i in xrange(means.shape[0]):
            plt.plot(means[i,:])
        plt.grid()
        plt.savefig("means.png")
        plt.figure()
        for i in xrange(newmeans.shape[0]):
            plt.plot(newmeans[i,:])
        x = time.time()
        print "time = %d" % x
        plt.title("@ %d" % x)
        plt.grid()
        plt.savefig("newmeans.png")
        scipy.io.savemat(meansfile, {'means' : newmeans})
        self.means = newmeans
        dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(means))
        print "dist:", dist.shape
        plt.figure()
        plt.imshow(dist, interpolation='nearest')
        plt.colorbar()
        plt.savefig('dist.png')
    if threaded:
        results = multithreading_pool_map(values, getMean)



class MeanCalculator(object):
    def __init__(self):
        self.enable_multithreading()

    def set_granules(self, granules):
        self._granules = granules
    def get_granules(self):
        return self._granules

    def set_number_of_groups(self, number_of_groups):
        self._number_of_groups = number_of_groups
    def get_number_of_groups(self):
        return self._number_of_groups
    def set_number_of_subgroups(self, number_of_subgroups):
        self._number_of_subgroups = number_of_subgroups
    def get_number_of_subgroups(self):
        return self._number_of_subgroups

    def set_number_of_runs(self, number_of_runs):
        self._number_of_runs = number_of_runs
    def get_number_of_runs(self):
        return self._number_of_runs

    def set_number_of_random_unique_subsamples(self, number_of_random_unique_subsamples):
        self._number_of_random_unique_subsamples = number_of_random_unique_subsamples
    def get_number_of_random_unique_subsamples(self):
        return self._number_of_random_unique_subsamples

    def set_number_of_observations(self, number_of_observations):
        self._number_of_observations = number_of_observations
    def get_number_of_observations(self):
        return self._number_of_observations

    def set_threshold(self, threshold):
        self._threshold = threshold
    def get_threshold(self):
        return self._threshold

    def set_labels(self, labels):
        self._labels = labels
    def get_labels(self):
        return self._labels

    def enable_multithreading(self):
        self._multithreading = True
    def disable_multithreading(self):
        self._multithreading = False
    def is_multithreading(self):
        return self._multithreading

    def set_number_of_points_mean_shift(self, number_of_points):
        self._number_of_points = number_of_points
    def get_number_of_points_mean_shift(self):
        return self._number_of_points

    def set_number_of_dimensions_mean_shift(self, number_of_dimensions):
        self._number_of_dimensions = number_of_dimensions
    def get_number_of_dimensions_mean_shift(self):
        return self._number_of_dimensions

    def set_number_of_neighbors_mean_shift(self, number_of_neighbors):
        self._number_of_neighbors = number_of_neighbors
    def get_number_of_neighbors_mean_shift(self):
        return self._number_of_neighbors

    def check_all_stats(self):
        assert self.get_number_of_random_unique_subsamples() and \
               self.get_number_of_groups() and \
               self.get_number_of_subgroups() and \
               self.get_number_of_runs() and \
               self.get_number_of_observations() and \
               self.get_granules() and \
               self.get_threshold() and \
               self.get_number_of_points_mean_shift() and \
               self.get_number_of_dimensions_mean_shift() and\
               self.get_number_of_neighbors_mean_shift()


    def set_means(self, means):
        self._means = means
    def get_means(self):
        return self._means

    def get_properties_as_array_dict(self):
        values = []
        for file in self.get_granules():
            values.append({})
            values[-1]['hdf_file'] = file
            values[-1]['number_of_runs'] = self.get_number_of_runs()
            values[-1]['number_of_observations'] = self.get_number_of_observations()
            values[-1]['number_of_random_unique_subsamples'] = self.get_number_of_random_unique_subsamples()
            values[-1]['threshold'] = self.get_threshold()

            values[-1]['number_of_points'] = self.get_number_of_points_mean_shift()
            values[-1]['number_of_dimensions'] = self.get_number_of_dimensions_mean_shift()
            values[-1]['number_of_neighbors'] = self.get_number_of_neighbors_mean_shift()
        return values

    def calculate_labels(self, temp_folder):
        self.check_all_properties()
        self.set_means(load_cached_or_calculate_and_cached('initial_means.obj', calc_means, self.get_properties_as_array_dict(), self.is_multithreading()))






