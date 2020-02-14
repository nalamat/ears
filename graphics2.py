'''GPU accelerated graphics and plotting using PyOpenGL.


This file is part of the EARS project: https://github.com/nalamat/ears
Copyright (C) 2017-2020 Nima Alamatsaz <nima.alamatsaz@gmail.com>
Copyright (C) 2017-2020 NESH Lab <ears.software@gmail.com>
Distrubeted under GNU GPLv3. See LICENSE.txt for more info.
'''

import math
import time
import logging
import functools
import threading
import numpy      as     np
import scipy      as     sp
import vispy      as     vp
import datetime   as     dt
from   scipy      import signal
from   vispy      import app, gloo, visuals
from   PyQt5      import QtCore, QtGui, QtWidgets
from   ctypes     import sizeof, c_void_p, c_bool, c_uint, c_float
from   OpenGL.GL  import *

import misc
import pipeline

log = logging.getLogger(__name__)
sizeFloat = sizeof(c_float)

_glslTransform = vp.visuals.shaders.Function('''
    vec2 transform(vec2 p, vec2 a, vec2 b) {
        return a*p + b;
    }

    vec2 transform(vec2 p, float ax, float ay, float bx, float by) {
        return transform(p, vec2(ax, ay), vec2(bx, by));
    }
    ''')

_glslRand = vp.visuals.shaders.Function('''
    float rand(vec2 seed){
        return fract(sin(dot(seed.xy ,vec2(12.9898,78.233))) * 43758.5453);
    }
    ''')

_defaultColors = np.array([
    [0 , 0,  .3],    # 1
    [0 , 0 , .6],    # 2
    [0 , 0 , 1 ],    # 3
    [0 , .3, 1 ],    # 4
    [0 , .6, 1 ],    # 5
    [0 , .6, .6],    # 6
    [0 , .3, 0 ],    # 7
    [0 , .6, 0 ],    # 8
    [0 , 1 , 0 ],    # 9
    [.5, .6, 0 ],    # 10
    [1 , .6, 0 ],    # 11
    [.3, 0 , 0 ],    # 12
    [.6, 0 , 0 ],    # 13
    [1 , 0 , 0 ],    # 14
    [1 , 0 , .3],    # 15
    [1 , 0 , .6],    # 16
    [.6, 0 , .6],    # 17
    [.6, 0 , 1 ],    # 18
    ])
