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

        self._required_properties = ['number_of_groups', 'number_of_subgroups', 'number_of_runs', 'number_of_random_unique_sub_samples',
                                     'number_of_observations', 'threshold', 'mean_shift']


    @property
    def granules(self):
        return self._granules
    @property
    def number_of_groups(self):
        return self._number_of_groups
    @property
    def number_of_subgroups(self):
        return self._number_of_subgroups
    @property
    def number_of_runs(self):
        return self._number_of_runs
    @property
    def number_of_random_unique_sub_samples(self):
        return self._number_of_random_unique_sub_samples
    @property
    def number_of_observations(self):
        return self._number_of_observations
    @property
    def threshold(self):
        return self._threshold

    @property
    def labels(self):
        return self._labels

    @property
    def mean_shift(self):
        return self._mean_shift
    @property
    def means(self):
        return self._means


    def enable_multithreading(self):
        self._multithreading = True
    def disable_multithreading(self):
        self._multithreading = False
    def is_multithreading(self):
        return self._multithreading



    def check_all_properties(self):
        for property in self._required_properties:
            assert getattr(self, property)



    def get_properties_as_array_dict(self):
        values = []
        for file in self.granules:
            values.append({})
            values[-1]['hdf_file'] = file
            for property in self._required_properties:
                values[-1][property] = getattr(self, property)

        return values

    def calculate_labels(self, temp_folder):
        self.check_all_properties()
        self.means = load_cached_or_calculate_and_cached('initial_means.obj', calc_means, self.get_properties_as_array_dict(), self.is_multithreading())






