__author__ = 'samyvilar'

from labelling import get_labels
import numpy


data = numpy.random.rand(1000, 4)
means = numpy.random.rand(4, 4)

dist = numpy.zeros((data.shape[0], means.shape[0]))
for i in xrange(means.shape[0]):
    dist[:, i] = numpy.sum((data - means[i,:])**2, axis = 1)
labels = dist.argmin(axis = 1)

