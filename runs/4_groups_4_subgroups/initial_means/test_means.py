__author__ = 'Samy Vilar'
__date__ = 'Mar 22, 2012'

import sys
sys.path.extend('../../..')

from lookuptable.lookuptable import  lookuptable

lut = lookuptable()
lut.load_table('../lookuptable/reflectance/800/800_lookuptable.numpy')


