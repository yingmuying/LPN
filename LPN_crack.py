#-*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import backend as K
import numpy as np
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.preprocessing import image
import keras.callbacks
import string , random
import opt
from optparse import OptionParser



'''
parameters
'''
parser = OptionParser()
OUTPUT_DIR = 'data'
(options, args) = parser.parse_args()


'''
license plate string geanerator
'''





