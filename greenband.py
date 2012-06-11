'''
Authors: Samy Vilar with contributions from
         Irina Gladkova, Fazlul Shahriar, George Bonev

Date:    January 5, 2010
'''

__author__ = "Samy Vilar"

import execnet
import networkx
import scipy.io
import scipy.optimize
import scipy.cluster.vq
import scipy.spatial.distance
import time
import numpy
import os
import os.path
import multiprocessing
import glasslab_cluster.cluster
import glasslab_cluster.io
import glasslab_cluster.cluster.consensus as gcons
import matplotlib.pyplot as plt
import pickle
import sys

import ann
import lookuptable

logging = 1
cpus = multiprocessing.cpu_count()

if cpus == 8:
    ncpus = 10
else:
    ncpus = cpus

def freelinuxRam():
    return int(os.popen("free -m").readlines()[1].split()[3])

class prop(object): # Used to move values to and from functions as a single object (JSON)
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
    def __eq__(self, other):
        return self.__dict__ == other.__dict__

# PREPROCESS ###################################################################
NDestripeBins   = 100
CropOrigin      = None
CropSize        = None
WinSize         = (5, 5)
FillWinSize     = 11
NSens           = 20
NNewMeans       = 10
PredictBand     = 4
ImgShape        = (2030, 1354)
NClusters       = 10
NRuns           = 10
NClSample       = 1000
NCoSample       = int(numpy.prod(ImgShape)*0.1)

def paddata(data, winsize):
    r = winsize[0]//2
    c = winsize[1]//2
    if r != 0:
        north = data[:r,:]
        south = data[-r:,:]
        data = numpy.concatenate((north[::-1,:], data, south[::-1,:]), axis = 0)
    if c != 0:
        east = data[:,:c]
        west = data[:,-c:]
        data = numpy.concatenate((east[:,::-1], data, west[:,::-1]), axis = 1)
    return data
def unpaddata(data, winsize):
    r = winsize[0]//2
    c = winsize[1]//2
    if r == 0 and c == 0:
        return data
    if r == 0:
        return data[:, c:-c]
    if c == 0:
        return data[r:-r, :]
    return data[r:-r, c:-c]
def fillinvalid(img, validrange):
    maxpad = FillWinSize//2
    maxinvalid = 0.50    # percent
    invalid = (validrange[0] > img) | (img > validrange[1])
    newinvalid = invalid.copy()
    newimg = img.copy()
    # Don't try to fill pixels near the border
    fillable = invalid.copy()
    fillable[:maxpad,:], fillable[-maxpad:,:] = False, False
    fillable[:,:maxpad], fillable[:,-maxpad:] = False, False
    for i, j in zip(*numpy.where(fillable)):
        for p in xrange(1, maxpad + 1):
            ind = numpy.s_[i-p:i+p+1, j-p:j+p+1]
            win = img[ind]
            wininv = invalid[ind]
            if numpy.sum(wininv)/(2.0*p+1.0)**2 < maxinvalid:
                newimg[i,j] = win[~wininv].mean()
                newinvalid[i,j] = False
                break
    return newimg, newinvalid
def destripe(im, nbins):
    destriped = im.copy()
    goodsen = 0
    lastrow = (im.shape[0]//NSens)*NSens
    strippedsens = range(1, 20)
    D1 = im[goodsen:lastrow:NSens, :]
    m = D1.min()
    M = D1.max()
    H = numpy.zeros((2, nbins))
    for sen in strippedsens:
        D       = im[sen:lastrow:NSens, :]
        x       = numpy.linspace(min(m,numpy.min(D)), max(M,numpy.max(D)), nbins+1)
        WH, _   = numpy.histogram(D1.ravel(), bins=x, normed=True)
        H[0,:]  = numpy.cumsum(WH)
        WH, _   = numpy.histogram(D.ravel(), bins=x, normed=True)
        H[1,:]  = numpy.cumsum(WH)
        y       = numpy.zeros(nbins)
        for i in xrange(nbins):
            indL = numpy.where(H[0,:] <= H[1,i])[0]
            if len(indL) == 0 or len(indL) == nbins:
                y[i] = x[i]
            else:
                pos = indL.max()
                xL = x[pos]
                fL = H[0, pos]
                xR = x[pos+1]
                fR = H[0, pos+1]
                y[i] = xL + (H[1,i]-fL)*(xR-xL)/float(fR-fL)
        Dall    = im[sen::NSens, :]
        B       = numpy.interp(Dall.ravel(), x[:-1], y)
        WH, _   = numpy.histogram(B, bins=x, normed=True)
        destriped[sen::NSens, :] = B.reshape(Dall.shape)
    return destriped
def read_mod02HKM(path, corig, csize, param, Bands, fillinvalidrange = False):
    data        = []
    scales      = numpy.zeros(len(Bands))
    offsets     = numpy.zeros(len(Bands))
    validrange  = numpy.zeros((len(Bands), 2))
    for i, band in enumerate(Bands):
        mod = glasslab_cluster.io.modis_level1b_read(path, band, param = param, clean = True)

        if numpy.any(mod['mimage'].mask):
            print "flags exist in band %d of granule %s" % (band, os.path.basename(path))
            raise Exception('Bad Granule')
        scales[i]       = mod['scale']
        offsets[i]      = mod['offset']
        validrange[i,:] = mod['validrange']

        img     = numpy.asarray(mod['mimage'].data, dtype = 'float').copy()

        '''
        img     = paddata(img.copy(), (FillWinSize, FillWinSize))
        img     = fillinvalid(img, validrange[i,:])[0]
        img     = unpaddata(img, (FillWinSize, FillWinSize))
        '''

        ################################################################################
        if fillinvalidrange:
            minind = (img < validrange[i, 0])
            maxind = (img > validrange[i, 1])
            if numpy.any(minind):
                img[minind] = validrange[i, 0]
                print "read_mod02HKM: granule: " + path + ", band: " + str(band) + ", pixels below valid range: " + str(numpy.sum(minind)) + " set to: " + str(validrange[i, 0])
            if numpy.any(maxind):
                img[maxind] = validrange[i, 1] - .01
                print "read_mod02HKM: granule: " + path + ", band: " + str(band) + ", pixels above valid range: " + str(numpy.sum(maxind)) + " set to: " + str(validrange[i, 1])
            ################################################################################

        #img     = destripe(img, NDestripeBins)

        n       = numpy.sum((validrange[i,0] > img) | (img > validrange[i,1]))
        if n > 0:
            print "Valid Range:", mod['validrange'], " %d values out of valid range in band %d" % (n, band)
            raise Exception('Bad Granule')

        if param == 'reflectance': #(path, band, mimage, param='radiance', start=None)
            os.system("cp " + path + " " + "/tmp/")
            tempfile = "/tmp/" + os.path.basename(path)
            glasslab_cluster.io.modis_level1b_write(tempfile, band, img, param = param)
            img = glasslab_cluster.io.modis_crefl(tempfile, bands = [band,])[0]
            os.system("rm " + "/tmp/" + os.path.basename(path))

        data.append(img)
    data = numpy.dstack(data)
    if corig and csize:
        crop = data[corig[0]:corig[0]+csize[0], corig[1]:corig[1]+csize[1]]
    else:
        crop = data
    return crop.reshape(crop.shape[0]*crop.shape[1], len(Bands)).astype(numpy.dtype('f8')), validrange
    #return data, crop, scales, offsets, validrange
################################################################################


def cheby_poly(points, degree):
    assert degree >= 0 and degree <= 9
    if   degree == 0:   return numpy.ones(len(points))
    elif degree == 1:   return     points
    elif degree == 2:   return 2*  points**2 - 1
    elif degree == 3:   return 4*  points**3 - 3*  points
    elif degree == 4:   return 8*  points**4 - 8*  points**2 + 1
    elif degree == 5:   return 16* points**5 - 20* points**3 + 5*  points
    elif degree == 6:   return 32* points**6 - 48* points**4 + 18* points**2 - 1
    elif degree == 7:   return 64* points**7 - 112*points**5 + 56* points**3 - 7*  points
    elif degree == 8:   return 128*points**8 - 256*points**6 + 160*points**4 - 32* points**2 + 1
    elif degree == 9:   return 256*points**9 - 576*points**7 + 432*points**5 - 120*points**3 + 9*points


# normalize each column x, y, z between -1 and 1, compute Cn,m,k 3D array which is replacing the look up tabel
# loop, ranging from 0..9, each is a product, find the dot product of the value of the function and
# the dot product if f(x,y,z) dot Tnmz

# f(x,y,z) dot T(x)T(y)T(z)

# make it an open interval on -1 and 1 => (-1, 1)

# N is the number of points,

def normalize(values, delta = .1):
    temp = numpy.zeros(values.shape)
    mins = []
    maxs = []
    for x in xrange(values.shape[1]):
        min = numpy.min(values[:, x]) - delta
        mins.append(min)
        max = numpy.max(values[:, x]) + delta
        maxs.append(max)
        temp[:, x]  = 2 * ((values[:, x] - min)/(max - min)) - 1
    return (temp, numpy.asarray(mins), numpy.asarray(maxs))

def calc_c(lut, m, n, k):
    Tx = cheby_poly(lut[:, 0], m)/(1 - lut[:, 0]**2)**.5
    Ty = cheby_poly(lut[:, 1], n)/(1 - lut[:, 1]**2)**.5
    Tz = cheby_poly(lut[:, 2], k)/(1 - lut[:, 2]**2)**.5
    return numpy.sum(lut[:, 3] * Tx * Ty * Tz)


def calc_coeff(lut, dim = 10):
    coeff = numpy.zeros((dim, dim, dim))
    normalize_lut, min, max = normalize(lut)
    normalize_lut[:, 3] = lut[:, 3]

    for n in xrange(dim):
        for m in xrange(dim):
            for k in xrange(dim):
                coeff[n, m , k] = calc_c(normalize_lut, n, m, k)

    return (coeff, min, max)

def restore_green_coeff(granule_dir, coef_table, min, max):
    hdffile = HDFFile(prop(file = granule_dir, bands = GreenBand.bands, param = 'radiance', load = True))
    data = hdffile.data

    for x in xrange(3):
        data[:, x] = 2 * ((data[:, x] - min[x])/(max[x] - min[x])) - 1

    assert data[:, x].min() >= -1 and data[:, x].max() <= 1

    green = numpy.zeros(data.shape[0])
    for n in xrange(coef_table.shape[0]):
        for m in xrange(coef_table.shape[1]):
            for k in xrange(coef_table.shape[2]):
                Tx = cheby_poly(data[:, 0], m)
                Ty = cheby_poly(data[:, 1], n)
                Tz = cheby_poly(data[:, 2], k)

                green = green + Tx * Ty * Tz * coef_table[n, m, k]
    return green


class HDFFile:
    file        = None
    bands       = None
    data        = None
    validrange  = None
    param       = None
    csize       = None
    corig       = None
    def __init__(self, prop):
        self.loadProperties(prop)
    def loadProperties(self, prop):
        if  hasattr(prop, 'file'):
            if os.path.isfile(prop.file):
                self.file = prop.file
            else:
                raise IOError("File: '" + prop.file + "' doesn't exist")
        if hasattr(prop, 'bands'):
            self.bands = prop.bands
        if hasattr(prop, 'data'):
            self.data = numpy.asarray(prop.data)
        if hasattr(prop, 'param'):
            self.param = prop.param
        if hasattr(prop, 'csize'):
            self.csize = prop.csize
        if hasattr(prop, 'corig'):
            self.corig = prop.corig
        if hasattr(prop, 'load') and prop.load:
            self.loadData()
    def loadData(self):
        if self.file == None:
            raise Exception('HDFFile object wasnt set with a file name ...')
        if self.bands == None:
            raise Exception('HDFFile object wasnt set with a set of bands ...')
        if self.param == None:
            raise Exception('HDFFile object wasnt set with param ...')
        try:
            self.data, self.validrange = read_mod02HKM(self.file, self.corig, self.csize, param = self.param, Bands = self.bands)
            if logging > 0:
                print self.file + ' has being loaded.'
        except Exception as inst:
            raise inst
    def nbytes(self):
        return (sys.getsizeof(self) + numpy.asarray(self.bands).nbytes + self.data.nbytes + sys.getsizeof(self.file))

def getMeans(data, labels):
    assert data.ndim == 2 and labels.ndim == 1
    assert data.shape[0] == len(labels)
    assert labels.min() >= 0
    nclust = labels.max() + 1
    means = numpy.zeros((nclust, data.shape[1]), dtype='f8')
    count = numpy.zeros(nclust, dtype='i')
    for i in xrange(nclust):
        ind = numpy.where(labels==i)[0]
        means[i,:] =  data[ind,:].mean(axis=0)
        count[i] = len(ind)
    return means, count

def meanshift(data):
    K = 30
    L = 1
    k = 100
    f = glasslab_cluster.cluster.FAMS(data, seed=100)
    #K, L = f.FindKL(10, 30, 2, 46, k, 0.05)
    #print 'K is', K
    #print 'L is', L

    pilot = f.RunFAMS(K, L, k)
    modes = f.GetModes()
    umodes = glasslab_cluster.utils.uniquerows(modes)
    labels = numpy.zeros(modes.shape[0])
    for i, m in enumerate(umodes):
        labels[numpy.all(modes==m, axis=1)] = i
    return umodes, labels, pilot

def merge_cluster(pattern, lbl_composites):
    try:
        pattern.shape #test if pattern is a NUMPY array, convert if list
    except:
        pattern = numpy.array( pattern )
        #print "lbl_composites:", lbl_composites
    for i, composite in enumerate(lbl_composites):
        for label in composite:
            if label != i:
                pattern[numpy.where(pattern == label)] = i

    return pattern

'''
def getMean(hdffile):
    data = hdffile.data
    if numpy.any(data.min(axis=0) == data.max(axis=0)):
        print "Skipping %s" % basename(path)# competive learning will crash, so lets skip it
        return None
    if logging > 0:
        print 'data shape:', data.shape, data.min(), data.max()
    assert numpy.all(numpy.isfinite(data))
    def clfunc(data):
        return glasslab_cluster.cluster.competlearn(data, NClusters, eta = 0.8)
    def cofunc(rlabels):
        return gcons.BestOfK(rlabels)
    def clproc(data):
        time.sleep(1)   # change seed for competlearn
        return scipy.cluster.vq.whiten(data - data.mean(axis = 0))
    rlabels, _ = gcons.subsampled(data, NRuns,
                cofunc=cofunc, clfunc=clfunc,
                nco=NCoSample, ncl=NClSample, clproc=clproc)
    print 'majority rule...'
    mrlabels = gcons.rmajrule(rlabels)
    means, _ = getMeans(data, mrlabels)
    return means
'''

def getMean(hdffile):
    data = hdffile.data
    if numpy.any(data.min(axis=0) == data.max(axis=0)):
        print "Skipping %s" % basename(path)# competive learning will crash, so lets skip it
        return None
    if logging > 0:
        print 'data shape:', data.shape, data.min(), data.max()
    assert numpy.all(numpy.isfinite(data))
    def clfunc(data):
        thresh = 0.8
        means, sublabels, pilot = meanshift(data)
        print 'means.shape' + str(means.shape)
        dmat = scipy.spatial.distance.pdist(means)
        print "dmat min max:", dmat.min(), dmat.max()
        dmat[dmat > thresh] = 0
        H = networkx.from_numpy_matrix(scipy.spatial.distance.squareform(dmat))
        cc = networkx.connected_components(H)
        print len(cc), "components:", map(len, cc)
        labels = merge_cluster(sublabels, cc) # modify in order  to merge means ...
        return labels
    def cofunc(rlabels):
        return gcons.BestOfK(rlabels)
    def clproc(data):
        time.sleep(1)   # change seed for competlearn
        return scipy.cluster.vq.whiten(data - data.mean(axis = 0))
    rlabels, _ = gcons.subsampled(data, NRuns,
        cofunc=cofunc, clfunc=clfunc,
        nco=NCoSample, ncl=NClSample, clproc=clproc)
    print 'majority rule...'
    mrlabels = gcons.rmajrule(rlabels)
    means, _ = getMeans(data, mrlabels)
    return means

def func(props):
    hdffile = HDFFile(props)
    try:
        hdffile.loadData()
        return hdffile
    except Exception as inst:
        return None

def appendones(matrix, axis = 1):
    return numpy.append(matrix, numpy.ones((matrix.shape[0], 1)), axis = axis)

class Lstsq(object):
    def __init__(self):
        self.Ag     = 0
        self.AA     = 0
        self.Nold   = 0
        self.Nnew   = 0
    def update(self, A, g):
        if len(A) == 0 and len(g) == 0:
            return
        self.Nnew   = self.Nold + A.shape[0]
        self.AA     = (1 / float(self.Nnew)) * (numpy.dot(A.T, A) + self.Nold * self.AA)
        self.Ag     = (1 / float(self.Nnew)) * (numpy.dot(A.T, g) + self.Nold * self.Ag)
        self.Nold   = self.Nnew
    def result(self):
        return numpy.dot(numpy.linalg.inv(self.AA), self.Ag)

def nnlabel(data, means):
    dist = numpy.zeros((data.shape[0], means.shape[0]))
    for i in xrange(means.shape[0]):
        dist[:,i] = numpy.sum((data - means[i,:])**2, axis=1)
    return dist.argmin(axis=1)

def calcAlpha(nc):
    predictind  = numpy.where(GreenBand.bands == PredictBand)[0]
    trainind    = numpy.where(GreenBand.bands != PredictBand)[0]
    ls          = Lstsq()
    for file in greenband.hdffiles:
        c = file.data[file.labels == nc, :]
        ls.update(appendones(c[:,trainind]), c[:,predictind])
    return ls.result()
def calcLabel(hdffile):
    return nnlabel(hdffile.data, hdffile.means)
def getAlphas(means):
    ############################################################################
    for hdffile in greenband.hdffiles:
        hdffile.means = means
    pool   = multiprocessing.Pool(processes = ncpus)
    labels = pool.map(calcLabel, greenband.hdffiles)
    pool.close()
    pool.join()
    index = 0
    for label in labels:
        greenband.hdffiles[index].labels = label
        index = index + 1
        ############################################################################
    try:
        pool   = multiprocessing.Pool(processes = ncpus)
        alphas = pool.map(calcAlpha, xrange(means.shape[0]))
        pool.close()
        pool.join()
    except:
        global allmeans
        global deltas
        global allalphas
        scipy.io.savemat("meanopt_" + str(len(deltas)) + ".mat",
                {
                "means"  : allmeans,
                "deltas" : deltas,
                "alphas" : allalphas,
                })
        print "There was an error ... current mean is " + str(means)

    pool.close()
    pool.join()
    return numpy.column_stack(alphas).T



def calcAlphaDist(nc, data):
    predictind  = numpy.where(GreenBand.bands == PredictBand)[0]
    trainind    = numpy.where(GreenBand.bands != PredictBand)[0]
    ls          = Lstsq()
    for file in greenband.hdffiles:
        c = file.data[file.labels == nc, :]
        ls.update(appendones(c[:,trainind]), c[:,predictind])
    return ls.result()
def getAlphasDist(means):
    ############################################################################
    for file in greenband.hdffiles:
        file.means = means
    group = execnet.Group(['popen'] * len(greenband.hdffiles))
    ch = []
    cmd = 'from greenband import *; channel.send(pickle.dumps(calcLabel(pickle.loads(channel.receive())), pickle.HIGHEST_PROTOCOL)'
    for index in xrange(len(greenband.hdffiles)):
        ch.append(group[index].remote_exec(cmd))
        ch[index].send(greenband.hdffiles[index])

    labels = []
    for index in xrange(len(ch)):
        labels.append(ch[index].receive())
        ############################################################################

    ch = []
    for index in xrange(xrange(numpy.asarray(means).shape[0])):
        ch.append(group[index].remote_exec('from greenband import *; channel.send()'))
        pool   = multiprocessing.Pool(processes = ncpus)
        alphas = pool.map(calcAlpha, xrange(means.shape[0]))
        pool.close()
        pool.join()

    pool.close()
    pool.join()
    return numpy.column_stack(alphas).T


class GreenBand(object):
    bands       = numpy.array([1,  2,  3,  4], dtype = 'i') # Bands to work with
    hdffiles    = []    # list of verified hdf files including their absolute location
    dir         = None
    means       = []
    files       = None
    meanopt     = None
    lookuptable = None
    def getPreProcessedFileDN(self, dir, totalnumbtoload = None):
        if totalnumbtoload == None:
            return dir + 'hdffile.obj'
        else:
            return dir + 'hdffile_' + str(totalnumbtoload) + '.obj'
    def loadPreProcessed(self, dir, totalnumbtoload = None):
        start   = time.time()
        self.hdffiles = pickle.load(open(self.getPreProcessedFileDN(dir, totalnumbtoload), 'r'))
        print "GreenBand(setGranulesDir): total number of files loaded: " + str(len(self.hdffiles))
        print "GreenBand(setGranulesDir) total time for loading: " + str(int((time.time() - start)))
    def setGranulesDir(self, dir, totalnumbtoload = None, corig = None, csize = None, param = None):
        print "GreenBand(setGranulesDir): Looking for preprocessed Granules ..."
        if os.path.isfile(self.getPreProcessedFileDN(dir, totalnumbtoload)):
            self.dir = dir
            self.loadPreProcessed(dir, totalnumbtoload)
            return
        self.dir = dir
        print "GreenBand(setGranulesDir): Looking for files within '" + dir + "'"
        files = os.listdir(dir)
        print "GreenBand(setGranulesDir): " + str(len(files)) + " files found. "
        hdffiles = []
        for file in files:
            if os.path.splitext(file)[1] == ".hdf":
                hdffiles.append(prop(file = dir + file, bands = GreenBand.bands, param = param, corig = corig, csize = csize))
        print "GreenBand(setGranulesDir): " + str(len(hdffiles)) + ' .hdf founds.'

        start   = time.time()
        pool    = multiprocessing.Pool(processes = ncpus)
        results = pool.map(func, hdffiles)
        pool.close()
        pool.join()
        for value in results:
            if isinstance(value, HDFFile):
                self.hdffiles.append(value)
                if len(self.hdffiles) == totalnumbtoload:
                    break
        print "GreenBand(setGranulesDir) done total time: " + str(int((time.time() - start)/60))
        print "GreenBand(setGranulesDir): total number of files preprocessed: " + str(len(self.hdffiles))
        print "GreenBand(setGranulesDir): saving preprocessed granules to disk ..."
        pickle.dump(self.hdffiles, open(self.getPreProcessedFileDN(dir, totalnumbtoload), 'w'), pickle.HIGHEST_PROTOCOL)
        print 'done ...'
    def getmeans(self, data, labels):
        nclust = labels.max()+1
        mu = numpy.zeros((nclust, data.shape[1]))
        for i in xrange(nclust):
            mu[i,:] = data[labels==i].mean(axis=0)
        return mu
    def save_reducedmeans(self, means, meansfile):
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
    def loadInitialMeans(self, meansfile):
        if os.path.isfile(meansfile):
            print "GreenBand(loadInitialMeans): " + meansfile + " mean file found"
            self.means = scipy.io.loadmat(meansfile)['means']
            print "GreenBand(loadInitialMeans): means shape: " + str(self.means.shape)
        else:
            print "GreenBand(loadInitialMeans): calculating means ..."
            start       = time.time()
            pool        = multiprocessing.Pool(processes = ncpus)
            means    = pool.map(getMean, self.hdffiles)
            pool.close()
            pool.join()
            print "GreenBand(loadInitialMeans) done total time: " + str(int((time.time() - start)/60))
            self.save_reducedmeans(means, meansfile)
    def findOptMean(self, iters = 1050):
        if logging > 0:
            print 'findOptMean: Starting ...'
            print 'findOptMean: iters: ' + str(iters)
        start = time.time()
        meanopt = scipy.optimize.fmin(localfit, greenband.means.ravel(), maxiter=iters, maxfun=iters)[0]
        print 'findOptMean(): done total time ' + str((time.time() - start)/60) + 'm'
        print 'saving results: deltas, means, alphas ...'

        global allmeans
        global deltas
        global allalphas
        scipy.io.savemat("meanopt_" + str(len(deltas)) + ".mat",
                {
                "means"  : allmeans,
                "deltas" : deltas,
                "alphas" : allalphas,
                })
    def save_figs(self, orig, pred):
        global error
        #original fig
        plt.figure()
        pmin = min(pred.min(), orig.min())
        pmax = max(pred.max(), orig.max())
        plt.imshow(orig, interpolation = 'nearest', vmin = pmin, vmax = pmax)
        plt.colorbar()
        global testfile
        temp = testfile.file.split('/')
        if len(temp) > 1:
            filename = temp[len(temp) - 1]
        else:
            filename = temp[0]
        plt.title("orig band4 "+ filename)
        plt.savefig("orig_granBand4_" + filename + ".png")
        #predicted fig
        plt.figure()
        plt.imshow(pred, interpolation = 'nearest', vmin = pmin, vmax = pmax)
        plt.colorbar()
        plt.title("pred band4 " + filename)
        plt.savefig("pred_granBand4_" + filename + ".png")
        #error
        plt.figure()
        error = numpy.fabs(orig - pred)
        #raise Exception("purposeful crash")
        plt.imshow(error, interpolation = 'nearest')
        plt.colorbar()
        plt.title("abs error for " + filename)
        plt.savefig("error_" + filename + ".png")
        #hist
        plt.figure()
        plt.hist(error.ravel(), bins = 100)
        plt.title("hist of abs error for " + filename)
        plt.savefig("hist_error_"+ filename + ".png")
    def savefigures(self):
        global predictind
        global trainind
        data = testfile.data
        original = data[:,predictind].reshape(ImgShape)
        predicted = numpy.zeros(data.shape[0])
        global allmeans
        global deltas
        index = numpy.argmin(deltas)
        means = allmeans[index]
        global allalphas
        alphas = allalphas[index]
        labels = nnlabel(data, means)
        for i in xrange(means.shape[0]):
            predicted[labels == i] = numpy.dot(appendones(data[labels == i,:][:,trainind]), alphas[i,:].reshape((-1,1)))
        predicted.shape = ImgShape
        original = data[:,predictind].reshape(ImgShape)
        plt.plot(deltas)
        plt.savefig('localfit.png')
        self.save_figs(original, predicted)
        predicted = predicted.astype('float32')
        predicted.tofile('predicted.dat')
        print numpy.mean(100*(original-predicted)/original)

    def loadLUT(self, lutfile, lutsize = 200):
        if os.path.isfile(lutfile):
            self.lookuptable = pickle.load(open(lutfile, 'r'))
        else:
            if logging > 0:
                print 'Creating look up table ...'
                #pool = multiprocessing.Pool(processes = 8)
            #start = time.time()
            #results = pool.map(getlookuptable, self.hdffiles)
            #pool.close()
            #pool.join()
            #end = time.time()
            lut = numpy.zeros((lutsize, lutsize, lutsize), dtype = 'int')
            counts = numpy.zeros((lutsize, lutsize, lutsize), dtype = 'int')
            start = time.time()
            for hdffile in greenband.hdffiles:
                s, c = lookuptable.lookuptable(hdffile, lutsize = lutsize)
                lut += s
                counts += c
            end = time.time()

            if logging > 0:
                print 'done total time: ' + str(end - start) + 's'
                print 'Merging all lookuptables ...'

            print 'done ... saved to ' + lutfile
            indices = lut != 0
            lut[indices] /= counts[indices]
            self.lookuptable = lut
            pickle.dump(lut, open(lutfile, 'w'), pickle.HIGHEST_PROTOCOL)
            scipy.io.savemat(lutfile + '.mat', {'lut':lut})
    def predictLUT(self, hdffile, lutsize = 200):
        global predictind
        global trainind
        original = hdffile.data[:,predictind].reshape(ImgShape)

        predicted = lookuptable.apply(hdffile, self.lookuptable)
        '''
        predicted = numpy.zeros(hdffile.data.shape[0])
        index = 0
        for row in hdffile.data:
            temp = ((row[0:3]/testfile.validrange[0:3, 1]) * lutsize).astype('int')
            val = greenband.lookuptable[temp[0], temp[1], temp[2]]
            predicted[index] = (val/lutsize) * hdffile.validrange[predictind, 1]
            index += 1
        '''
        predicted.shape = ImgShape
        greenband.save_figs(original, predicted)

        predicted = predicted.astype('float32')
        predicted.tofile('predicted.dat')

        print numpy.mean(100*(original-predicted)/original)



def getlookuptable(hdffile):
    return lookuptable.lookuptable(hdffile)






deltas      = []
allmeans    = []
allalphas   = []

predictind  = numpy.where(GreenBand.bands == PredictBand)[0]
trainind    = numpy.where(GreenBand.bands != PredictBand)[0]
alphas      = None
labels      = None
def cal(i):
    global testfile
    global alphas
    global labels
    trainind    = numpy.where(GreenBand.bands != PredictBand)[0]
    return numpy.dot(appendones(testfile.data[labels == i, :][:, trainind]), alphas[i,:].reshape((-1,1)))
def localfit(meansx):
    start       = time.time()
    means       = meansx.reshape(10, len(GreenBand.bands))
    global allmeans
    allmeans.append(means)
    global alphas
    alphas      = numpy.asarray(getAlphas(means))
    global testfile
    predicted   = numpy.zeros(testfile.data.shape[0])
    nclust      = means.shape[0]
    global labels
    labels      = nnlabel(testfile.data, means)

    pool        = multiprocessing.Pool(processes = ncpus)
    results     = pool.map(cal, xrange(nclust))
    pool.close()
    pool.join()
    index = 0
    for value in results:
        predicted[labels == index] = value
        index = index + 1
    original = testfile.data[:, predictind]
    original = original.reshape(original.shape[0],)

    temp = numpy.sum((predicted - original)**2)
    global deltas
    deltas.append(temp)
    global allalphas
    allalphas.append(alphas)
    print "localfit iteration: " + str(len(deltas)) + " time: " + str(round((time.time() - start), 8)) + "s value: " + str(temp)
    return temp

class Cluster(object):
    machines = [{'host':'foucault', 'ncpus':10},
            {'host':'fermi',    'ncpus':10},
            {'host':'fermat',   'ncpus':4},
            {'host':'fresnel',  'ncpus':4},
            {'host':'flemming', 'ncpus':4},
            {'host':'fock',     'ncpus':4},
            {'host':'gauss',    'ncpus':4}]




def init(granules):
    global greenband
    greenband = GreenBand()
    group = execnet.Group(['popen'] * len(granules))
    ch = []
    cmd = 'from greenband import *; channel.send(pickle.dumps(HDFFile(file = channel.receive(), bands = GreenBand.bands, load = True), pickle.HIGHEST_PROTOCOL))'
    for index in xrange(len(granules)):
        temp = group[index].remote_exec(cmd)
        temp.send(granules[index])
        ch.append(temp)

    for index in xrange(len(granules)):
        greenband.hdffiles.append(pickle.loads(ch[index].receive()))

    '''
    greenband.hdffiles = pool.map(func, granules)
    pool.close()
    pool.join()
    for granule in granules:
        greenband.hdffiles.append(HDFFile(prop(file = granule, bands = GreenBand.bands, load = True)))
    '''
    group.terminate()
    return 'loaded all granules ...'


greenband = None

if __name__ == "__channelexec__":
    for item in channel:
        channel.send(eval(item))


