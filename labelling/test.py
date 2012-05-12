__author__ = 'samyvilar'

from labelling import get_labels
import numpy
import time
import scipy.cluster.vq

if __name__ == '__main__':
    data = numpy.random.rand(10000000, 4)
    means = numpy.random.rand(4, 4)

    start = time.time()
    dist = numpy.zeros((data.shape[0], means.shape[0]))
    for i in xrange(means.shape[0]):
        dist[:, i] = numpy.sum((data - means[i,:])**2, axis = 1)
    labels = dist.argmin(axis = 1)
    end = time.time()
    print "Numpy time %f" % (end - start)

    start = time.time()
    labels2 = get_labels(data = data, means = means)
    end = time.time()
    print "C time %f" % (end - start)

    assert all(labels == labels2)
    print "OK"


data = numpy.random.rand(10, 4)
means = numpy.random.rand(4, 4, 4)
start = time.time()
dist = numpy.zeros((data.shape[0], means.shape[0], means.shape[1]))
for mean_index, mean in enumerate(means):
    for i in xrange(mean.shape[0]):
        dist[:, mean_index, i] = numpy.sum((data - mean[i,:])**2, axis = 1)


    labels = numpy.zeros((data.shape[0], 2), dtype = 'int')
    labels[:, 0] = dist.sum(axis = 2).argmin(axis = 1)
    for index in xrange(dist.shape[0]):
        labels[index, 1] = dist[index][labels[index, 0]].argmin()


    end = time.time()
    print "2D Numpy time %f" % (end - start)

    start = time.time()
    labels2 = get_labels(data = data, means = means)
    end = time.time()
    print "2D time %f" % (end - start)

    assert numpy.sum(labels - labels2) == 0
    print "2D OK"


