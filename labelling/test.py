__author__ = 'samyvilar'

from labelling import get_labels
import numpy
import time

if __name__ == '__main__':
    data = numpy.random.rand(10000, 4)
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

