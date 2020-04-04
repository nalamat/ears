'''GPU accelerated graphics and plotting using PyOpenGL.


This file is part of the EARS project: https://github.com/nalamat/ears
Copyright (C) 2017-2020 Nima Alamatsaz <nima.alamatsaz@gmail.com>
Copyright (C) 2017-2020 NESH Lab <ears.software@gmail.com>
Distributed under GNU GPLv3. See LICENSE.txt for more info.
'''

# notes:
# - GL color range is 0-1, Qt color range is 0-255
# - Qt window dimensions are normalized by the device pixel ratio, i.e. a
#   800x600 window looks almost the same on both low and high DPI screens.
#   this causes an issue that texts drawn with QPainter without adjusting for
#   pixel ratio will look pixelated on high DPI.
# - OpenGL coordinates in fragment shader are actual screen pixel values and
#   must be normalized manually if necessary. Look at getPixelRatio().

import sys
import math
import time
import logging
import functools
import threading
import collections
import numpy       as     np
import scipy       as     sp
import datetime    as     dt
from   scipy       import signal
from   PyQt5       import QtCore, QtGui, QtWidgets
from   ctypes      import sizeof, c_void_p, c_bool, c_uint, c_float, string_at
from   OpenGL.GL   import *

import misc
import config
import pipeline


log = logging.getLogger(__name__)
sizeFloat = sizeof(c_float)

glVersion = (4, 1)
glSamples = 0
maxChannels = 128
max3DTextureSize = 1024


defaultColors = np.array([
    [0 , 0  , .3 , 1],    # 1
    [0 , 0  , .55, 1],    # 2
    [0 , 0  , .7 , 1],    # 3
    [0 , 0  , 1  , 1],    # 4
    [0 , .3 , 0  , 1],    # 5
    [0 , .45, 0  , 1],    # 6
    [0 , .6 , 0  , 1],    # 7
    [0 , .75, 0  , 1],    # 8
    [0 , .85, 0  , 1],    # 9
    [0 , 1  , 0  , 1],    # 10
    [.3, 0  , 0  , 1],    # 11
    [.6, 0  , 0  , 1],    # 12
    [1 , 0  , 0  , 1],    # 13
    [1 , .4 , 0  , 1],    # 14
    [1 , .6 , 0  , 1],    # 15
    [1 , .8 , 0  , 1],    # 16
    [.9, .9 , 0  , 1],    # 17
    [0 , .4 , .4 , 1],    # 18
    [0 , .6 , .6 , 1],    # 19
    [0 , .8 , .8 , 1],    # 20
    [.4, 0  , .4 , 1],    # 21
    [.7, 0  , .7 , 1],    # 22
    [1 , 0  , 1  , 1],    # 23
    [0 , 0  , 0  , 1],    # 24
    [.2, .2 , .2 , 1],    # 25
    [.4, .4 , .4 , 1],    # 26
    [.6, .6 , .6 , 1],    # 27
    [.8, .8 , .8 , 1],    # 28
    ], dtype=np.float32)


def getPixelRatio():
    '''Use for high DPI display support.
    For example, retina displays have a pixel ratio of 2.0.
    '''
    return QtWidgets.QApplication.screens()[0].devicePixelRatio()


class Program:
    '''OpenGL shader program.'''

    _helperFunctions = '''
        vec2 transform(vec2 p, vec2 a, vec2 b) {
            return a*p + b;
        }

        vec2 transform(vec2 p, float ax, float ay, float bx, float by) {
            return transform(p, vec2(ax, ay), vec2(bx, by));
        }

        float rand(vec2 seed) {
            return fract(sin(dot(seed.xy, vec2(12.9898,78.233))) * 43758.5453);
        }

        bool between(float a, float b, float c) {
            return a <= b && b < c;
        }

        bool between(int a, int b, int c) {
            return a <= b && b < c;
        }

        float sinFade(float fade) {
            return sin((fade*2-1)*3.1415/2)/2+.5;
        }
        '''

    @classmethod
    def createShader(cls, shaderType, source):
        """Compile a shader."""
        source = ('#version %d%d0 core\n' % glVersion) + \
            cls._helperFunctions + source
        shader = glCreateShader(shaderType)
        glShaderSource(shader, source)
        glCompileShader(shader)

        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(shader))

        return shader

    def __init__(self, vert=None, tesc=None, tese=None, geom=None, frag=None,
            comp=None):

        if (vert or tesc or tese or geom or frag) and not (vert and frag):
            raise ValueError('Require both vertex and fragment shaders')
        if (vert or tessc or tesse or geom or frag) and comp:
            raise ValueError('Compute shader cannot be linked with others')

        if vert: vert = Program.createShader(GL_VERTEX_SHADER         , vert)
        if tesc: tesc = Program.createShader(GL_TESS_CONTROL_SHADER   , tesc)
        if tese: tese = Program.createShader(GL_TESS_EVALUATION_SHADER, tese)
        if geom: geom = Program.createShader(GL_GEOMETRY_SHADER       , geom)
        if frag: frag = Program.createShader(GL_FRAGMENT_SHADER       , frag)
        if comp: comp = Program.createShader(GL_COMPUTE_SHADER        , comp)

        self.id = glCreateProgram()

        if vert: glAttachShader(self.id, vert)
        if tesc: glAttachShader(self.id, tesc)
        if tese: glAttachShader(self.id, tese)
        if geom: glAttachShader(self.id, geom)
        if frag: glAttachShader(self.id, frag)
        if comp: glAttachShader(self.id, comp)

        glLinkProgram(self.id)

        if glGetProgramiv(self.id, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(self.id))

        if vert: glDeleteShader(vert)
        if tesc: glDeleteShader(tesc)
        if tese: glDeleteShader(tese)
        if geom: glDeleteShader(geom)
        if frag: glDeleteShader(frag)
        if comp: glDeleteShader(comp)

        self.vao = glGenVertexArrays(1)
        self.vbos = {}
        self.ebo = None

        self._contexts = 0

    def __enter__(self):
        if not self._contexts:
            self.begin()
        self._contexts += 1

    def __exit__(self, *args):
        self._contexts -= 1
        if not self._contexts:
            self.end()

    def begin(self):
        glUseProgram(self.id)
        glBindVertexArray(self.vao)

    def end(self):
        glBindVertexArray(0)
        glUseProgram(0)

    def setUniform(self, name, type, value):
        with self:
            if not isinstance(value, collections.abc.Iterable):
                value = (value,)
            id = glGetUniformLocation(self.id, name)
            globals()['glUniform' + type](id, *value)

    def setVBO(self, name, type, size, data, usage):
        '''Copy vertex data to a VBO and link to its vertex attribute.

        Args:
            name (str): As defined in the shader.
            type (int): Of each component, e.g. GL_BYTE, GL_INT, GL_FLOAT
            size (int): Number of components per vertex.
            data (np.ndarray): Vertex data.
            usage (int): Expected usage pattern, e.g. GL_STATIC_DRAW,
                GL_DYNAMIC_DRAW, GL_STREAM_DRAW
        '''
        with self:
            if name not in self.vbos: self.vbos[name] = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbos[name])
            glBufferData(GL_ARRAY_BUFFER, data, usage)
            loc = glGetAttribLocation(self.id, name)
            glVertexAttribPointer(loc, size, type, GL_FALSE, 0, c_void_p(0))
            glEnableVertexAttribArray(loc)

    def subVBO(self, name, offset, data):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbos[name])
        glBufferSubData(GL_ARRAY_BUFFER, offset, data)

    def setEBO(self, data, usage):
        with self:
            if not self.ebo: self.ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, data, usage)


class Item:
    '''Graphical item capable of containing child items.'''

    @property
    def canvas(self):
        if self.parent:
            return self.parent.canvas
        else:
            return None

    def __init__(self, parent, **kwargs):
        '''
        Args:
            parent
            pos (2 floats): Item position in unit parent space (x, y).
            size (2 floats): Item size in unit parent space (w, h).
            margin (4 floats): Margin in pixels (l, t, r, b).
            bgColor (3 floats): Backgoround color.
            fgColor (3 floats): Foreground color for border and texts.
        '''

        # properties with their default values
        defaults = dict(pos=(0,0), size=(1,1), margin=(0,0,0,0),
            bgColor=(1,1,1,0), fgColor=(0,0,0,1), visible=True)

        self._initProps(defaults, kwargs)

        super().__init__()

        self._props['parent']  = parent
        self._props['posPxl']  = np.array([0, 0])
        self._props['sizePxl'] = np.array([1, 1])
        self._initialized = False
        self._items = []

        if isinstance(self.parent, Item):
            self.parent.addItem(self)

    def _initProps(self, defaults, kwargs):
        # initialize properties with the given or default values
        # setter monitoring (refreshing, etc) won't apply here
        if not hasattr(self, '_props'):
            self._props = dict()

        for name, default in defaults.items():
            if name in self._props:
                continue
            elif name in kwargs:
                self._props[name] = kwargs[name]
            else:
                self._props[name] = default

    def __getattr__(self, name):
        if name != '_props' and hasattr(self, '_props') and name in self._props:
            return self._props[name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name != '_props' and hasattr(self, '_props') and name in self._props:
            if name in {'parent', 'posPxl', 'sizePxl'}:
                raise AttributeError('Cannot set attribute')

            self._props[name] = value

            if name in {'pos', 'size', 'margin'}:
                self.resize()

        else:
            super().__setattr__(name, value)

    def addItem(self, item):
        if not isinstance(item, Item):
            raise TypeError('`item` should be an Item')

        self._items += [item]

    def callItems(self, func, *args, **kwargs):
        for item in self._items:
            if hasattr(item, func):
                getattr(item, func)(*args, **kwargs)

    # initally called from Canvas.initializeGL
    def initializeGL(self):
        self._initialized = True
        self.callItems('initializeGL')

    # initally called from Canvas.resizeGL
    def resizeGL(self):
        parent = self.parent
        margin = np.array(self.margin)

        self._props['posPxl'] = self.pos * parent.sizePxl + \
            parent.posPxl + margin[[0,3]]
        self._props['sizePxl'] = self.size * parent.sizePxl - \
            margin[0:2] - margin[2:4]

        self.callItems('resizeGL')

    # initally called from Canvas.paintGL
    def paintGL(self):
        self.callItems('paintGL')

    # functions chained to all child items
    _funcs = []
    for func in _funcs:
        exec('def %(func)s(self, *args, **kwargs):\n'
            '    self.callItems("%(func)s", *args, **kwargs)' % {'func':func})


class Text(Item):
    '''Graphical text item.'''

    _vertShader = '''
        in vec2 aVertex;
        in vec2 aTexCoord;

        out vec2 TexCoord;

        uniform vec2  uPos;         // unit
        uniform vec2  uAnchor;      // unit
        uniform vec2  uSize;        // pixels
        uniform vec2  uParentPos;   // pixels
        uniform vec2  uParentSize;  // pixels
        uniform vec2  uCanvasSize;  // pixels

        void main() {
            vec2 pos = (uPos * uParentSize + uParentPos) / uCanvasSize;
            vec2 size = uSize / uCanvasSize;

            gl_Position = vec4(
                ((aVertex - uAnchor) * size + pos) * 2 - vec2(1),
                0, 1);
            TexCoord = aTexCoord;
        }
        '''

    _fragShader = '''
        in vec2 TexCoord;

        out vec4 FragColor;

        uniform sampler2D uTexture;

        void main() {
            FragColor = texture(uTexture, TexCoord);
        }
        '''

    @classmethod
    def getSize(cls, text, font):
        '''Get width and height of the given text in pixels'''
        if not hasattr(cls, '_image'):
            cls._image = QtGui.QImage(1, 1, QtGui.QImage.Format_ARGB32)
            cls._painter = QtGui.QPainter()
            cls._painter.begin(cls._image)

        cls._painter.setFont(font)
        rect = cls._painter.boundingRect(QtCore.QRect(),
            QtCore.Qt.AlignLeft, text)

        return rect.width(), rect.height()

    def __init__(self, parent, **kwargs):
        '''
        Args:
            text (str)
            pos (2 floats): In unit parent space.
            anchor (2 floats): In unit text space.
            margin (4 ints): In pixels (l, t, r, b).
            fontSize (float)
            bold (bool)
            italic (bool)
            align (QtCore.Qt.Alignment): Possible values are AlignLeft,
                AlignRight, AlignCenter, AlignJustify.
            fgColor (4 floats): RGBA 0-1.
            bgColor (4 floats): RGBA 0-1.
            visible (bool)
        '''

        # properties with their default values
        defaults = dict(text='', pos=(.5,.5), anchor=(.5,.5), margin=(0,0,0,0),
            fontSize=12, bold=False, italic=False, align=QtCore.Qt.AlignCenter,
            bgColor=(1,1,1,0), fgColor=(0,0,0,1))

        self._initProps(defaults, kwargs)

        super().__init__(parent, **kwargs)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if not hasattr(self, '_initialized') or not self._initialized: return

        if name != '_uProps' and hasattr(self, '_uProps') and \
                name in self._uProps:
            self._prog.setUniform(*self._uProps[name], value)

        if name in {'text', 'margin', 'fontSize', 'bold', 'italic',
                'align', 'bgColor', 'fgColor'}:
            self.refresh()

    def initializeGL(self):
        vertices = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
            ], dtype=np.float32)

        indices = np.array([
            [0,1,3],
            [1,2,3],
            ], dtype=np.uint32)

        texCoords = np.array([
            [0, 1],
            [0, 0],
            [1, 0],
            [1, 1],
            ], dtype=np.float32)

        self._tex = glGenTextures(1)

        self._prog = Program(vert=self._vertShader, frag=self._fragShader)

        # properties linked with a uniform variable in the shader
        self._uProps = dict(pos=('uPos', '2f'), anchor=('uAnchor', '2f'))

        with self._prog:
            for name, value in self._uProps.items():
                self._prog.setUniform(*value, self._props[name])

            self._prog.setVBO('aVertex', GL_FLOAT, 2, vertices, GL_STATIC_DRAW)
            self._prog.setVBO('aTexCoord', GL_FLOAT, 2, texCoords,
                GL_STATIC_DRAW)
            self._prog.setEBO(indices, GL_STATIC_DRAW)

        self.refresh()

        super().initializeGL()

    def resizeGL(self):
        super().resizeGL()

        with self._prog:
            self._prog.setUniform('uParentPos', '2f', self.parent.posPxl)
            self._prog.setUniform('uParentSize', '2f', self.parent.sizePxl)
            self._prog.setUniform('uCanvasSize', '2f', self.canvas.sizePxl)

    def refresh(self):
        '''Generates texture for the given text, font, color, etc.
        Note: This function does not force redrawing of the text on screen.
        '''
        ratio = getPixelRatio()
        margin = (np.array(self.margin) * ratio).astype(np.int)
        font = QtGui.QFont('arial', self.fontSize * ratio,
            QtGui.QFont.Bold if self.bold else QtGui.QFont.Normal, self.italic)

        w, h = Text.getSize(self.text, font)
        w += margin[0] + margin[2]
        h += margin[1] + margin[3]
        if w==0: w += 1
        if h==0: h += 1

        image = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32)
        image.fill(QtGui.QColor(*np.array(self.bgColor)*255))

        painter = QtGui.QPainter()
        painter.begin(image)
        painter.setPen(QtGui.QColor(*np.array(self.fgColor)*255))
        painter.setFont(font)
        painter.drawText(margin[0], margin[1],
            w - margin[0] - margin[2], h - margin[1] - margin[3],
            self.align, self.text)
        painter.end()
        str = image.bits().asstring(w * h * 4)
        texData = np.frombuffer(str, dtype=np.uint8).reshape((h, w, 4))

        self._prog.setUniform('uSize', '2f', (w/ratio, h/ratio))

        # texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texData.shape[1],
            texData.shape[0], 0, GL_BGRA, GL_UNSIGNED_BYTE, texData)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        # glGenerateMipmap(GL_TEXTURE_2D)

    def paintGL(self):
        if not self.visible: return

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex)

        with self._prog:
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, c_void_p(0))

        super().paintGL()


class Rectangle(Item):
    '''Graphical rectangle with border and fill.'''

    _vertShader = '''
        in vec2 aVertex;

        uniform vec2  uPos;         // unit
        uniform vec2  uSize;        // unit
        uniform vec4  uMargin;      // pixels: l t r b
        uniform vec2  uParentPos;   // pixels
        uniform vec2  uParentSize;  // pixels
        uniform vec2  uCanvasSize;  // pixels

        void main() {
            // transform inside parent
            vec2 pos = (uPos * uParentSize + uParentPos) / uCanvasSize;
            vec2 size = uSize * uParentSize / uCanvasSize;

            // add margin
            pos += uMargin.xw / uCanvasSize;
            size -= (uMargin.xy + uMargin.zw) / uCanvasSize;

            // apply transformation to vertex and take to NDC space
            gl_Position = vec4((aVertex * size + pos) * 2 - vec2(1), 0, 1);
        }
        '''

    _fragShader = '''
        out vec4 FragColor;

        uniform vec2  uPos;         // unit
        uniform vec2  uSize;        // unit
        uniform vec4  uMargin;      // pixels: l t r b
        uniform float uBorderWidth; // pixels
        uniform vec2  uParentPos;   // pixels
        uniform vec2  uParentSize;  // pixels
        uniform vec2  uCanvasSize;  // pixels
        uniform float uPixelRatio;
        uniform vec4  uBgColor;
        uniform vec4  uFgColor;

        void main() {
            // high DPI display support
            vec2 fragCoord = gl_FragCoord.xy / uPixelRatio;

            // transform to pixel space
            vec4 rect = vec4(uPos * uParentSize + uParentPos + uMargin.xw,
                (uPos + uSize) * uParentSize + uParentPos - uMargin.zy).xwzy;

            // check border
            if (between(rect.x, fragCoord.x, rect.z) &&
                    (between(-uBorderWidth, fragCoord.y-rect.y, 0) ||
                    between(0, fragCoord.y-rect.w, uBorderWidth)) ||
                    between(rect.w, fragCoord.y, rect.y) &&
                    (between(0, fragCoord.x-rect.x, uBorderWidth) ||
                    between(-uBorderWidth, fragCoord.x-rect.z, 0)))
                FragColor = uFgColor;
            else
                FragColor = uBgColor;
        }
        '''

    def __init__(self, parent, **kwargs):
        '''
        '''

        # properties with their default values
        defaults = dict(borderWidth=1)

        self._initProps(defaults, kwargs)

        super().__init__(parent, **kwargs)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if not hasattr(self, '_initialized') or not self._initialized: return

        if name != '_uProps' and hasattr(self, '_uProps') and \
                name in self._uProps:
            self._prog.setUniform(*self._uProps[name], value)

    def initializeGL(self):
        vertices = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
            ], dtype=np.float32)

        self._indices = np.array([
            [0, 1, 2],
            [0, 2, 3],
            ], dtype=np.uint32)

        # properties linked with a uniform variable in the shader
        self._uProps = dict(pos=('uPos', '2f'), size=('uSize', '2f'),
            margin=('uMargin', '4f'), borderWidth=('uBorderWidth', '1f'),
            bgColor=('uBgColor', '4f'), fgColor=('uFgColor', '4f'))

        self._prog = Program(vert=self._vertShader, frag=self._fragShader)

        with self._prog:
            for name, value in self._uProps.items():
                self._prog.setUniform(*value, self._props[name])
            # high DPI display support
            self._prog.setUniform('uPixelRatio', '1f', getPixelRatio())

            self._prog.setVBO('aVertex', GL_FLOAT, 2, vertices, GL_STATIC_DRAW)
            self._prog.setEBO(self._indices, GL_STATIC_DRAW)

        super().initializeGL()

    def resizeGL(self):
        super().resizeGL()

        with self._prog:
            self._prog.setUniform('uParentPos', '2f', self.parent.posPxl)
            self._prog.setUniform('uParentSize', '2f', self.parent.sizePxl)
            self._prog.setUniform('uCanvasSize', '2f', self.canvas.sizePxl)

    def paintGL(self):
        if not self.visible: return

        with self._prog:
            glDrawElements(GL_TRIANGLES, self._indices.size,
                GL_UNSIGNED_INT, c_void_p(0))

        super().paintGL()


class Grid(Rectangle):
    MAX_TICKS = 100

    _fragShader = '''
        out vec4 FragColor;

        uniform vec2  uPos;         // unit
        uniform vec2  uSize;        // unit
        uniform vec4  uMargin;      // pixels: l t r b
        uniform float uBorderWidth; // pixels
        uniform vec2  uParentPos;   // pixels
        uniform vec2  uParentSize;  // pixels
        uniform vec2  uCanvasSize;  // pixels
        uniform float uPixelRatio;
        uniform vec4  uBgColor;
        uniform vec4  uFgColor;
        uniform int   uXTickCount;
        uniform float uXTicks[{MAX_TICKS}]; // unit
        uniform int   uYTickCount;
        uniform float uYTicks[{MAX_TICKS}]; // unit

        void main() {
            // high DPI display support
            vec2 fragCoord = gl_FragCoord.xy / uPixelRatio;

            // transform to pixel space
            vec4 rect = vec4(uPos * uParentSize + uParentPos + uMargin.xw,
                (uPos + uSize) * uParentSize + uParentPos - uMargin.zy).xwzy;

            // check border
            for (int i=0; i<uXTickCount; ++i)
                if (int(fragCoord.x) == int(uXTicks[i]*(rect.z-rect.x)+rect.x)
                        && int(fragCoord.y)%3 == 0) {
                    FragColor = uFgColor;
                    return;
                }

            for (int i=0; i<uYTickCount; ++i)
                if (int(fragCoord.y) == int(uYTicks[i]*(rect.y-rect.w)+rect.w)
                        && int(fragCoord.x)%3 == 0) {
                    FragColor = uFgColor;
                    return;
                }

            FragColor = vec4(0);
        }
        ''' \
        .replace('{MAX_TICKS}', str(MAX_TICKS))

    def __init__(self, parent, **kwargs):
        # properties with their default values
        defaults = dict(xTicks=[], yTicks=[])

        self._initProps(defaults, kwargs)

        super().__init__(parent, **kwargs)

    def __setattr__(self, name, value):
        # verify value
        if name in {'xTicks', 'yTicks'}:
            if len(value) > Grid.MAX_TICKS:
                raise ValueError('Tick count cannot exceed %d' % Grid.MAX_TICKS)

        # set attribute
        super().__setattr__(name, value)

        if not hasattr(self, '_initialized') or not self._initialized: return

        # set uniforms
        if name == 'xTicks':
            with self._prog:
                self._prog.setUniform('uXTickCount', '1i', len(self.xTicks))
                self._prog.setUniform('uXTicks', '1fv',
                    (len(self.xTicks), self.xTicks))

        elif name == 'yTicks':
            with self._prog:
                self._prog.setUniform('uYTickCount', '1i', len(self.yTicks))
                self._prog.setUniform('uYTicks', '1fv',
                    (len(self.yTicks), self.yTicks))

    def initializeGL(self):
        super().initializeGL()

        # set uniforms
        with self._prog:
            self.xTicks = self.xTicks
            self.yTicks = self.yTicks


class Plot(Item):
    def __init__(self, parent, **kwargs):
        # properties with their default values
        defaults = dict()

        self._initProps(defaults, kwargs)

        if not isinstance(parent, Figure):
            raise TypeError('Parent must be a Figure')
        self._props['figure'] = parent
        parent = parent.view

        super().__init__(parent, **kwargs)

    def __setattr__(self, name, value):
        if name in {'figure'}:
            raise AttributeError('Cannot set attribute')

        super().__setattr__(name, value)


class AnalogPlot(Plot, pipeline.Sampled):
    _vertShader = '''
        in vec2 aVertex;

        out vec2 vVertex;

        uniform vec2  uPos;         // unit
        uniform vec2  uSize;        // unit
        uniform vec4  uMargin;      // pixels: l t r b
        uniform vec2  uParentPos;   // pixels
        uniform vec2  uParentSize;  // pixels
        uniform vec2  uCanvasSize;  // pixels

        void main() {
            // transform inside parent
            vec2 pos = (uPos * uParentSize + uParentPos) / uCanvasSize;
            vec2 size = uSize * uParentSize / uCanvasSize;

            // add margin
            pos += uMargin.xw / uCanvasSize;
            size -= (uMargin.xy + uMargin.zw) / uCanvasSize;

            // apply transformation to vertex and take to NDC space
            gl_Position = vec4( (aVertex * size + pos) * 2 - vec2(1), 0, 1);

            vVertex = aVertex;
        }
        '''

    _fragShader = '''
        in vec2 vVertex;

        out vec4 FragColor;

        uniform int   uNs;
        uniform int   uNsRange;
        uniform vec2  uTexSize;     // unit
        uniform float uFadeRange;   // unit

        uniform sampler2D uTexture;

        void main() {
            FragColor = texture(uTexture, vVertex * uTexSize);

            float x = vVertex.x;
            float diff = x - float(uNs % uNsRange) / uNsRange;
            float fade = 1;
            if (between(0, diff + 1, uFadeRange))
                fade = (diff + 1) / uFadeRange;
            else if (0 < diff && uNs < uNsRange)
                fade = 0;
            else if (between(0, diff, uFadeRange))
                fade = diff / uFadeRange;
            fade = sinFade(fade);
            FragColor.a *= fade;
        }
        '''

    _vertShader2 = '''
        in float aVertex;

        out float fChannel;

        uniform int   uNsRange;
        uniform int   uChannels;

        void main() {
            int channel = gl_VertexID / uNsRange;
            vec2 vertex = vec2(float(gl_VertexID % uNsRange) / uNsRange,
                ((aVertex + 1) / 2 + uChannels - 1 - channel) / uChannels);

            vertex = vertex * 2 - vec2(1);

            gl_Position = vec4(vertex, 0, 1);
            fChannel = channel;
        }
        '''

    _fragShader2 = '''
        in float fChannel;
        out vec4 FragColor;

        uniform vec4  uColor[{MAX_CHANNELS}];

        void main() {
            if (0 < fract(fChannel)) discard;
            FragColor = uColor[int(fChannel) % {MAX_COLORS}];
        }
        ''' \
        .replace('{MAX_COLORS}', str(len(defaultColors))) \
        .replace('{MAX_CHANNELS}', str(maxChannels))

    def __init__(self, parent, **kwargs):
        '''
        '''

        # properties with their default values
        defaults = dict()

        self._initProps(defaults, kwargs)

        if not isinstance(parent, Scope):
            raise TypeError('Parent must be a Scope')

        super().__init__(parent, **kwargs)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if not hasattr(self, '_initialized') or not self._initialized: return

        if name != '_uProps' and hasattr(self, '_uProps') and \
                name in self._uProps:
            self._prog.setUniform(*self._uProps[name], value)

    def _configuring(self, params, sinkParams):
        super()._configuring(params, sinkParams)

        if params['channels'] > maxChannels:
            raise ValueError('Channel count cannot exceed %d' % maxChannels)

    def _configured(self, params, sinkParams):
        super()._configured(params, sinkParams)

        self._nsRange = int(np.ceil(self.figure.tsRange * self.fs))

        self._buffer = misc.CircularBuffer((self.channels, self._nsRange),
            dtype=np.float32)

    def _written(self, data, source):
        with self._buffer:
            self._buffer.write(data)

        super()._written(data, source)

    def initializeGL(self):
        vertices = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
            ], dtype=np.float32)

        indices = np.array([
            [0, 1, 3],
            [1, 2, 3],
            ], dtype=np.uint32)

        self._prog = Program(vert=self._vertShader, frag=self._fragShader)

        # properties linked with a uniform variable in the shader
        self._uProps = dict(pos=('uPos', '2f'), size=('uSize', '2f'),
            margin=('uMargin', '4f'))

        with self._prog:
            for name, value in self._uProps.items():
                self._prog.setUniform(*value, self._props[name])

            self._prog.setUniform('uNsRange', '1i', self._nsRange)
            self._prog.setUniform('uFadeRange', '1f', self.figure.fadeRange)

            self._prog.setVBO('aVertex', GL_FLOAT, 2, vertices, GL_STATIC_DRAW)
            self._prog.setEBO(indices, GL_STATIC_DRAW)

        self._fboSize = (4000, 4000)
        self._tex = glGenTextures(1)
        self._fbo = glGenFramebuffers(1)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, *self._fboSize, 0, GL_RGBA,
            GL_UNSIGNED_BYTE, c_void_p(0))
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D, self._tex, 0)
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Framebuffer not complete')

        self._prog2 = Program(vert=self._vertShader2, frag=self._fragShader2)

        with self._prog2:
            self._prog2.setUniform('uNsRange', '1i', np.int32(self._nsRange))
            self._prog2.setUniform('uChannels', '1i', np.int32(self.channels))
            self._prog2.setUniform('uColor', '4fv',
                (self.channels, defaultColors))

            self._prog2.setVBO('aVertex', GL_FLOAT, 1, self._buffer._data,
                GL_DYNAMIC_DRAW)

        self._labels = [None] * self.channels
        for channel in range(self.channels):
            self._labels[channel] = Text(parent=self, text=str(channel+1),
                pos=(0, 1-(channel+.5)/self.channels), anchor=(1,.5),
                margin=(0,0,5+self.margin[0],0), fontSize=18, bold=True,
                fgColor=defaultColors[channel % len(defaultColors)])

        self.refresh()

        super().initializeGL()

    def resizeGL(self):
        super().resizeGL()

        with self._prog:
            self._prog.setUniform('uParentPos', '2f', self.parent.posPxl)
            self._prog.setUniform('uParentSize', '2f', self.parent.sizePxl)
            self._prog.setUniform('uCanvasSize', '2f', self.canvas.sizePxl)
            self._prog.setUniform('uTexSize', '2f',
                (self.sizePxl / self._fboSize * getPixelRatio()) \
                .clip(None, [1,1]))

        self.refresh()

    def refresh(self, ns1=None, ns2=None):
        if ns1 is None: ns1 = 0
        if ns2 is None: ns2 = self._nsRange

        if ns1 == ns2:
            return
        # wrap around
        if ns1 > ns2:
            self.refresh(ns1, self._nsRange)
            self.refresh(0, ns2)
            return

        sizePxl = (self.sizePxl * getPixelRatio()).astype(np.int32)
        sizePxl = sizePxl.clip(None, self._fboSize)

        # save currently bound framebuffer and viewport
        fbo = glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING)
        viewport = glGetIntegerv(GL_VIEWPORT)

        # switch to the offscreen framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glViewport(0, 0, *sizePxl)

        # when drawing lines, have to connect the new samples to the last one
        ns1 = max(ns1 - 1, 0)

        # left and right most pixels of the redraw region
        x1 = int(np.floor(ns1 / self._nsRange * sizePxl[0]))
        x2 = int(np.ceil (ns2 / self._nsRange * sizePxl[0]))

        # align starting sample to the left most pixel of the redraw region
        ns1 = int(x1 / sizePxl[0] * self._nsRange)

        # clear and draw only the region that needs to be updated
        glEnable(GL_SCISSOR_TEST)
        glScissor(x1, 0, x2 - x1, sizePxl[1])

        glClearColor(*self.bgColor)
        glClear(GL_COLOR_BUFFER_BIT)

        # draw the analog signals
        with self._prog2:
            for i in range(self.channels):
                glDrawArrays(GL_LINE_STRIP, i * self._nsRange + ns1, ns2 - ns1)

        glDisable(GL_SCISSOR_TEST)

        # restore the previously bound framebuffer and viewport
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glViewport(*viewport)

    def _subVBO(self, ns1, ns2, data):
        '''Update a subregion of VBO'''
        # nothing to update
        if ns1 == ns2:
            return
        # wrap around
        if ns1 > ns2:
            self._subVBO(ns1, self._nsRange, data[:, :self._nsRange-ns1])
            self._subVBO(0, ns2, data[:, self._nsRange-ns1:])
            return

        # transfer samples to GPU channel by channel
        with self._prog2:
            for i in range(self.channels):
                self._prog2.subVBO('aVertex',
                    (i * self._nsRange + ns1) * sizeFloat,
                    data[i, :])

    def paintGL(self):
        if not self.visible: return

        with self._buffer:
            if self._buffer.nsAvailable:
                ns1 = self._buffer.nsRead % self._nsRange
                ns2 = self._buffer.nsWritten % self._nsRange
                data = self._buffer.read()
                self._subVBO(ns1, ns2, data)
                self.refresh(ns1, ns2)
                self._prog.setUniform('uNs', '1i', self._buffer.nsWritten)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex)

        with self._prog:
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, c_void_p(0))

        super().paintGL()


class EpochPlot(Plot, pipeline.Node):
    _vertShader = '''
        in float aData;

        uniform vec2  uPos;         // unit
        uniform vec2  uSize;        // unit
        uniform vec4  uMargin;      // pixels: l t r b
        uniform vec2  uParentPos;   // pixels
        uniform vec2  uParentSize;  // pixels
        uniform vec2  uCanvasSize;  // pixels

        uniform float uTs;
        uniform float uTsRange;

        void main() {
            // transform inside parent
            vec2 pos = (uPos * uParentSize + uParentPos) / uCanvasSize;
            vec2 size = uSize * uParentSize / uCanvasSize;

            // add margin
            pos += uMargin.xw / uCanvasSize;
            size -= (uMargin.xy + uMargin.zw) / uCanvasSize;

            float ts = 0 <= aData && aData <= uTs ? aData : uTs;

            pos.x += (ts / uTsRange - int(uTs / uTsRange)) * size.x;
            pos.y += size.y / 2;

            // apply transformation to vertex and take to NDC space
            gl_Position = vec4(pos * 2 - vec2(1), 0, 1);
        }
        '''

    _geomShader = '''
        layout (lines) in;
        layout (triangle_strip, max_vertices = 8) out;

        out float gPos;

        uniform vec2  uPos;         // unit
        uniform vec2  uSize;        // unit
        uniform vec4  uMargin;      // pixels: l t r b
        uniform vec2  uParentPos;   // pixels
        uniform vec2  uParentSize;  // pixels
        uniform vec2  uCanvasSize;  // pixels

        uniform float uTs;
        uniform float uTsRange;

        void emitRectangle(vec4 pos0, vec4 pos1, vec2 pos, vec2 size) {
            gl_Position = pos0 + vec4(0, +size.y, 0, 0);
            gPos = ((pos0.x + 1) / 2 - pos.x) / size.x;
            EmitVertex();

            gl_Position = pos0 + vec4(0, -size.y, 0, 0);
            gPos = ((pos0.x + 1) / 2 - pos.x) / size.x;
            EmitVertex();

            gl_Position = pos1 + vec4(0, +size.y, 0, 0);
            gPos = ((pos1.x + 1) / 2 - pos.x) / size.x;
            EmitVertex();

            gl_Position = pos1 + vec4(0, -size.y, 0, 0);
            gPos = ((pos1.x + 1) / 2 - pos.x) / size.x;
            EmitVertex();

            EndPrimitive();
        }

        void main() {
            // transform inside parent
            vec2 pos = (uPos * uParentSize + uParentPos) / uCanvasSize;
            vec2 size = uSize * uParentSize / uCanvasSize;

            // add margin
            pos += uMargin.xw / uCanvasSize;
            size -= (uMargin.xy + uMargin.zw) / uCanvasSize;

            // left and right bounds (NDC)
            float left = pos.x * 2 - 1;
            float right = (pos.x + size.x) * 2 - 1;

            float posTs = pos.x +
                (uTs / uTsRange - int(uTs / uTsRange)) * size.x;
            posTs = posTs * 2 - 1;

            // main epoch
            vec4 pos0 = gl_in[0].gl_Position;
            vec4 pos1 = gl_in[1].gl_Position;

            if (pos1.x <= posTs - size.x * 2) return;

            if (pos1.x >= left) {
                if (pos0.x < left) pos0.x = left;

                emitRectangle(pos0, pos1, pos, size);
            }

            // ghost epoch
            pos0 = gl_in[0].gl_Position;
            pos1 = gl_in[1].gl_Position;

            if (pos0.x < left) {
                pos0.x += size.x * 2;
                pos1.x += size.x * 2;

                if (pos0.x < posTs) pos0.x = posTs;
                if (right < pos1.x) pos1.x = right;

                emitRectangle(pos0, pos1, pos, size);
            }
        }
        '''

    _fragShader = '''
        in float gPos;

        out vec4 FragColor;

        uniform vec2  uPos;         // unit
        uniform vec2  uSize;        // unit
        uniform vec4  uMargin;      // pixels: l t r b
        uniform vec2  uParentPos;   // pixels
        uniform vec2  uParentSize;  // pixels
        uniform vec2  uCanvasSize;  // pixels
        uniform float uPixelRatio;
        uniform vec4  uFgColor;

        uniform float uTs;
        uniform float uTsRange;
        uniform float uFadeRange;   // unit

        void main() {
            // high DPI display support
            //vec2 fragCoord = gl_FragCoord.xy / uPixelRatio;

            // transform to pixel space
            //vec4 rect = vec4(uPos * uParentSize + uParentPos + uMargin.xw,
            //    (uPos + uSize) * uParentSize + uParentPos - uMargin.zy).xwzy;

            FragColor = uFgColor;

            float diff = gPos - (uTs / uTsRange - int(uTs / uTsRange));
            float fade = 1;
            if (between(0, diff + 1, uFadeRange))
                fade = (diff + 1) / uFadeRange;
            else if (between(0, diff, uFadeRange))
                fade = diff / uFadeRange;
            fade = sinFade(fade);
            FragColor.a *= fade;
        }
        '''

    @property
    def aux(self):
        return self._aux

    def __init__(self, parent, **kwargs):
        defaults = dict(name='', fgColor=(0,0,1,.6))

        self._initProps(defaults, kwargs)

        super().__init__(parent, **kwargs)

        self._aux = pipeline.Auxillary()
        self._cacheSize = 100*2    # cache 100 epochs
        self._pointerWrite = 0     # pointer to last element written in _data
        self._pointerRead = 0      # pointer to last element read from _data
        self._partial = None
        self._data = np.zeros(self._cacheSize, dtype=np.float32)
        self._lock = threading.Lock()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if not hasattr(self, '_initialized') or not self._initialized: return

        if name != '_uProps' and hasattr(self, '_uProps') and \
                name in self._uProps:
            self._prog.setUniform(*self._uProps[name], value)
        elif name == 'name' and hasattr(self, '_label'):
            self._label.text = value

    def _writing(self, data, source):
        data = super()._writing(data, source)

        if not isinstance(data, np.ndarray): data = np.array(data)
        if data.ndim == 0: data = data[np.newaxis]
        if data.ndim != 1: raise ValueError('`data` should be 1D')

        return data

    def _written(self, data, source):
        with self._lock:
            # when last added epoch has been partial, complete the epoch by
            # adding its start timestamp to the beginning of current data, and
            # also rewind the read and write pointers
            if self._partial is not None:
                self._pointerWrite -= 2
                self._pointerRead = min(self._pointerRead, self._pointerWrite)
                data = np.insert(data, 0, self._partial)
                self._partial = None

            # when last epoch in the current data is partial, save its start
            # timestamp for the next write operation (see above)
            if len(data) % 2 != 0:
                self._partial = data[-1]
                data = np.append(data, -1)

            # write data to (circular) buffer
            window = np.arange(self._pointerWrite,
                self._pointerWrite + len(data))
            window %= self._cacheSize
            self._data[window] = data

            self._pointerWrite += len(data)

        super()._written(data, source)

    def initializeGL(self):
        # properties linked with a uniform variable in the shader
        self._uProps = dict(pos=('uPos', '2f'), size=('uSize', '2f'),
            margin=('uMargin', '4f'), fgColor=('uFgColor', '4f'))

        self._prog = Program(vert=self._vertShader,
            geom=self._geomShader,
            frag=self._fragShader)

        with self._prog:
            for name, value in self._uProps.items():
                self._prog.setUniform(*value, self._props[name])
            # high DPI display support
            self._prog.setUniform('uPixelRatio', '1f', getPixelRatio())
            self._prog.setUniform('uTsRange', '1f', self.figure.tsRange)
            self._prog.setUniform('uFadeRange', '1f', self.figure.fadeRange)

            self._prog.setVBO('aData', GL_FLOAT, 1, self._data, GL_DYNAMIC_DRAW)

        self._label = Text(self, text=self.name, pos=(0,.5), anchor=(1,.5),
                margin=(0,0,5+self.margin[0],0), fontSize=14, bold=True,
                fgColor=self.fgColor)

        super().initializeGL()

    def resizeGL(self):
        super().resizeGL()

        with self._prog:
            self._prog.setUniform('uParentPos', '2f', self.parent.posPxl)
            self._prog.setUniform('uParentSize', '2f', self.parent.sizePxl)
            self._prog.setUniform('uCanvasSize', '2f', self.canvas.sizePxl)

    def _subVBO(self, frm, to, data):
        '''Update a subregion of VBO'''
        # nothing to update
        if frm == to:
            return
        # wrap around
        if frm > to:
            self._subVBO(frm, len(data), data)
            self._subVBO(0, to, data)
            return

        self._prog.subVBO('aData', frm * sizeFloat, data[frm:to])

    def paintGL(self):
        if not self.visible: return

        with self._prog:
            # update subregion of VBO with the newly written epochs
            with self._lock:
                if self._pointerRead != self._pointerWrite:
                    if self._pointerWrite-self._pointerRead >= self._cacheSize:
                        self._subVBO(0, self._cacheSize, self._data)
                    else:
                        self._subVBO(self._pointerRead % self._cacheSize,
                            self._pointerWrite % self._cacheSize, self._data)
                    self._pointerRead = self._pointerWrite

            self._prog.setUniform('uTs', '1f', self.aux.ts)

            glDrawArrays(GL_LINES, 0, self._data.size)

        super().paintGL()


class SpikePlot(Plot, pipeline.Node):
    _vertShader = '''
        in float aData;

        out float vSpikeIndex;

        uniform vec2  uPos;         // unit
        uniform vec2  uSize;        // unit
        uniform vec4  uMargin;      // pixels: l t r b
        uniform vec2  uParentPos;   // pixels
        uniform vec2  uParentSize;  // pixels
        uniform vec2  uCanvasSize;  // pixels

        uniform int   uSpikeCount;
        uniform int   uSpikeLength;
        uniform int   uSpikeIndex;

        void main() {
            // transform inside parent
            vec2 pos = (uPos * uParentSize + uParentPos) / uCanvasSize;
            vec2 size = uSize * uParentSize / uCanvasSize;

            // add margin
            pos += uMargin.xw / uCanvasSize;
            size -= (uMargin.xy + uMargin.zw) / uCanvasSize;

            vSpikeIndex = gl_VertexID / uSpikeLength;
            float sampleIndex = gl_VertexID % uSpikeLength;
            vec2 vertex = vec2(sampleIndex / (uSpikeLength - 1),
                (aData + 1) / 2);

            vertex = (vertex * size + pos) * 2 - vec2(1);

            gl_Position = vec4(vertex, 0, 1);
        }
        '''

    _fragShader = '''
        in float vSpikeIndex;

        out vec4 FragColor;

        uniform vec2  uPos;         // unit
        uniform vec2  uSize;        // unit
        uniform vec4  uMargin;      // pixels: l t r b
        uniform vec2  uParentPos;   // pixels
        uniform vec2  uParentSize;  // pixels
        uniform vec2  uCanvasSize;  // pixels
        uniform float uPixelRatio;

        uniform int   uSpikeCount;
        uniform int   uSpikeIndex;
        uniform vec4  uFgColor;

        void main() {
            if (0 < fract(vSpikeIndex)) discard;
            if (uSpikeIndex <= vSpikeIndex) discard;

            // high DPI display support
            vec2 fragCoord = gl_FragCoord.xy / uPixelRatio;

            // transform to pixel space
            vec4 rect = vec4(uPos * uParentSize + uParentPos + uMargin.xw,
                (uPos + uSize) * uParentSize + uParentPos - uMargin.zy).xwzy;

            if (fragCoord.y > rect.y || rect.w > fragCoord.y) discard;

            FragColor = uFgColor;
            FragColor.a *= 1 - mod(uSpikeIndex - 1 - vSpikeIndex,
                uSpikeCount) / uSpikeCount;
        }
        '''

    def __init__(self, parent, **kwargs):
        defaults = dict(name='')

        self._initProps(defaults, kwargs)

        super().__init__(parent, **kwargs)

        self._spikeCount = 20
        self._pointerWrite = 0
        self._pointerRead = 0
        self._lock = threading.Lock()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if not hasattr(self, '_initialized') or not self._initialized: return

        if name != '_uProps' and hasattr(self, '_uProps') and \
                name in self._uProps:
            self._prog.setUniform(*self._uProps[name], value)
        elif name == 'name' and hasattr(self, '_label'):
            self._label.text = value

    def _configuring(self, params, sinkParams):
        super()._configuring(params, sinkParams)

        if 'spikeLength' not in params:
            raise ValueError('SpikePlot requires `spikeLength`')

        if 'channels' in params and params['channels'] != 1:
            raise ValueError('SpikePlot can only plot a single `channel`')

    def _configured(self, params, sinkParams):
        super()._configured(params, sinkParams)

        self._spikeLength = params['spikeLength']
        self._data = np.zeros((self._spikeCount, self._spikeLength),
            dtype=np.float32)

    def _writing(self, data, source):
        data = super()._writing(data, source)

        for ts, peak, spike in data:
            if len(spike) != self._spikeLength:
                raise ValueError('Spike length discrepancy (%d != %d)' %
                    (len(spike), self._spikeLength))

        return data

    def _written(self, data, source):
        with self._lock:
            for ts, peak, spike in data:
                self._data[self._pointerWrite % self._spikeCount, :] = spike
                self._pointerWrite += 1

        super()._written(data, source)

    def initializeGL(self):
        # properties linked with a uniform variable in the shader
        self._uProps = dict(pos=('uPos', '2f'), size=('uSize', '2f'),
            margin=('uMargin', '4f'), fgColor=('uFgColor', '4f'))

        self._prog = Program(vert=self._vertShader, frag=self._fragShader)

        with self._prog:
            for name, value in self._uProps.items():
                self._prog.setUniform(*value, self._props[name])
            # high DPI display support
            self._prog.setUniform('uPixelRatio', '1f', getPixelRatio())
            self._prog.setUniform('uSpikeLength', '1i', self._spikeLength)
            self._prog.setUniform('uSpikeCount', '1i', self._spikeCount)
            self._prog.setUniform('uSpikeIndex', '1i', 0)

            self._prog.setVBO('aData', GL_FLOAT, 1, self._data, GL_DYNAMIC_DRAW)

        self._label = Text(self, text=self.name, pos=(0,1), margin=(4,4,0,0),
            anchor=(0,1), fontSize=14, bold=True, fgColor=self.fgColor)

        super().initializeGL()

    def resizeGL(self):
        super().resizeGL()

        with self._prog:
            self._prog.setUniform('uParentPos', '2f', self.parent.posPxl)
            self._prog.setUniform('uParentSize', '2f', self.parent.sizePxl)
            self._prog.setUniform('uCanvasSize', '2f', self.canvas.sizePxl)

    def _subVBO(self, frm, to, data):
        '''Update a subregion of VBO'''
        # nothing to update
        if frm == to:
            return
        # wrap around
        if frm > to:
            self._subVBO(frm, self._spikeCount, data)
            self._subVBO(0, to, data)
            return

        self._prog.subVBO('aData', frm * self._spikeLength * sizeFloat,
            data[frm:to, :])

    def paintGL(self):
        if not self.visible: return

        with self._prog:
            # update subregion of VBO with the newly written spikes
            with self._lock:
                if self._pointerRead != self._pointerWrite:
                    if self._pointerWrite-self._pointerRead >= self._spikeCount:
                        self._subVBO(0, self._spikeCount, self._data)
                    else:
                        self._subVBO(self._pointerRead % self._spikeCount,
                            self._pointerWrite % self._spikeCount, self._data)
                    self._pointerRead = self._pointerWrite
                    self._prog.setUniform('uSpikeIndex', '1i',
                        self._pointerWrite)

            glDrawArrays(GL_LINE_STRIP, 0, self._data.size)

        super().paintGL()


class SpikeOverlay(Plot, pipeline.Node):
    _vertShader = '''
        in vec2 aVertex;

        out vec2 vVertex;

        uniform vec2  uPos;         // unit
        uniform vec2  uSize;        // unit
        uniform vec4  uMargin;      // pixels: l t r b
        uniform vec2  uParentPos;   // pixels
        uniform vec2  uParentSize;  // pixels
        uniform vec2  uCanvasSize;  // pixels

        void main() {
            // transform inside parent
            vec2 pos = (uPos * uParentSize + uParentPos) / uCanvasSize;
            vec2 size = uSize * uParentSize / uCanvasSize;

            // add margin
            pos += uMargin.xw / uCanvasSize;
            size -= (uMargin.xy + uMargin.zw) / uCanvasSize;

            // apply transformation to vertex and take to NDC space
            gl_Position = vec4( (aVertex * size + pos) * 2 - vec2(1), 0, 1);

            vVertex = aVertex;
        }
        '''

    _fragShader = '''
        out vec4 FragColor;

        uniform vec4 uFgColor;
        uniform float uMarkerSize;

        vec2 point;
        float point2;
        float pxl;

        void main() {
            // NDC point coordinate (-1 to +1)
            point = 2 * (gl_PointCoord - .5);
            point2 = dot(point, point);
            pxl = 2 / uMarkerSize;

            // full square
            float a = 1;

            // hollow square
            //a = step(1 - pxl, abs(point.y)) - step(1, abs(point.y)) +
            //    step(1 - pxl, abs(point.x)) - step(1, abs(point.x));
            //a = clamp(a, 0, 1);

            // +
            a = step(-pxl/2, point.y) - step(pxl/2, point.y) +
                step(-pxl/2, point.x) - step(pxl/2, point.x);
            a = clamp(a, 0, 1);

            // x
            a = smoothstep(-pxl, 0, abs(point.y) - abs(point.x)) -
                smoothstep(0, pxl, abs(point.y) - abs(point.x));

            // full circle
            // note: there's no performance difference with step vs smoothstep
            float r0 = 1;
            float r1 = 1 - 1 * pxl;
            //a = 1 - smoothstep(r1 * r1, r0 * r0, point2);

            // hollow circle
            float r2 = 1 - 2 * pxl;
            //a = smoothstep(r2 * r2, r1 * r1, point2) -
            //    smoothstep(r1 * r1, r0 * r0, point2);

            // down triangle
            //a = 1 - smoothstep(-pxl, 0, point.y + abs(point.x * 2) - 1);

            // up triangle
            //a = smoothstep(0, pxl, point.y - abs(point.x * 2) + 1);

            // left triangle
            //a = smoothstep(0, pxl, point.x - abs(point.y * 2) + 1);

            // right triangle
            //a = 1 - smoothstep(-pxl, 0, point.x + abs(point.y * 2) - 1);

            FragColor = uFgColor * vec4(1, 1, 1, a);
        }
        '''

    def __init__(self, parent, **kwargs):
        defaults = dict(markerSize=20)

        self._initProps(defaults, kwargs)

        super().__init__(parent, **kwargs)

        self._channels = None
        self._pointSize = self.markerSize * getPixelRatio()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if not hasattr(self, '_initialized') or not self._initialized: return

        if name == 'markerSize':
            self._pointSize = value * getPixelRatio()

        if name != '_uProps' and hasattr(self, '_uProps') and \
                name in self._uProps:
            self._prog.setUniform(*self._uProps[name], value)

    def _configuring(self, params, sinkParams):
        super()._configuring(params, sinkParams)

        if 'channels' not in params:
            raise ValueError('SpikeOverlay requires `channels`')

    def _configured(self, params, sinkParams):
        super()._configured(params, sinkParams)

        self._channels = params['channels']

    def initializeGL(self):
        self._vertices = np.array([
            [.5, .5],
            [.25, .25],
            [.25, .75],
            [.75, .25],
            [.75, .75],
            ], dtype=np.float32)

        # self._vertices = np.random.uniform(0,1,(100000,2)).astype(np.float32)

        self._prog = Program(vert=self._vertShader, frag=self._fragShader)

        # properties linked with a uniform variable in the shader
        self._uProps = dict(pos=('uPos', '2f'), size=('uSize', '2f'),
            margin=('uMargin', '4f'), markerSize=('uMarkerSize', '1f'),
            fgColor=('uFgColor', '4f'))

        with self._prog:
            for name, value in self._uProps.items():
                self._prog.setUniform(*value, self._props[name])

            self._prog.setVBO('aVertex', GL_FLOAT, 2, self._vertices,
                GL_STATIC_DRAW)

        super().initializeGL()

    def resizeGL(self):
        super().resizeGL()

        with self._prog:
            self._prog.setUniform('uParentPos', '2f', self.parent.posPxl)
            self._prog.setUniform('uParentSize', '2f', self.parent.sizePxl)
            self._prog.setUniform('uCanvasSize', '2f', self.canvas.sizePxl)

    def paintGL(self):
        glPointSize(self._pointSize)
        with self._prog:
            glDrawArrays(GL_POINTS, 0, self._vertices.shape[0])

        super().paintGL()


class Figure(Item):
    def __init__(self, parent, **kwargs):
        '''
        '''

        # properties with their default values
        defaults = dict(borderWidth=1, zoom=(1,1), pan=(0,0),
            xTicks=[], yTicks=[])

        self._initProps(defaults, kwargs)

        super().__init__(parent, **kwargs)

        # view holds all plots
        self._props['view'] = Item(parent=self, pos=(0,0), size=(1,1),
            margin=(self.borderWidth,)*4)

        self._grid = Grid(parent=self.view,
            fgColor=self.fgColor*np.array([1,1,1,.2]),
            xTicks=self.xTicks, yTicks=self.yTicks)

        # a border around the figure
        self._border = Rectangle(parent=self, pos=(0,0), size=(1,1),
            borderWidth=self.borderWidth, bgColor=self.bgColor,
            fgColor=self.fgColor)

    def __setattr__(self, name, value):
        if name in {'view'}:
            raise AttributeError('Cannot set attribute')

        super().__setattr__(name, value)

        if name in {'xTicks', 'yTicks'}:
            setattr(self._grid, name, value)

    def _pxl2nrmView(self, pxlDelta):
        # translate a delta (dx, dy) from pixels to normalized in view space
        # mainly used for panning
        return pxlDelta * 2 / self._pxlViewSize * [1, -1]

    def on_mouse_wheel(self, event):
        if not self.isInside(event.pos): return

        scale = event.delta[0] if event.delta[0] else event.delta[1]
        scale = math.exp(scale * .15)
        if 'Control' in event.modifiers:    # both x and y zoom
            scale = np.array([scale, scale])
        elif 'Shift' in event.modifiers:
            scale = np.array([scale, 1])    # only x zoom
        elif 'Alt' in event.modifiers:
            scale = np.array([1, scale])    # only y zoom
        else:
            return                          # no zoom

        zoom = self.zoom
        self.zoom = zoom * scale
        scale = self.zoom / zoom
        delta = self._pxl2nrmView(event.pos - self._pxlViewCenter)
        self.pan = delta * (1 - scale) + self.pan * scale

        super().on_mouse_wheel(self, event)

    def on_mouse_press(self, event):
        if not self.isInside(event.pos): return

        if (event.button == 1 and 'Control' in event.modifiers):
            # start pan
            event.figure = self
            event.pan    = self.pan

    def on_mouse_move(self, event):
        if (event.press_event and hasattr(event.press_event, 'figure')
                and event.press_event.figure == self
                and event.press_event.button == 1
                and 'Control' in event.press_event.modifiers):
            # pan
            delta = self._pxl2nrmView(event.pos - event.press_event.pos)
            self.pan = event.press_event.pan + delta

    def on_mouse_release(self, event):
        if (event.press_event and hasattr(event.press_event, 'figure')
                and event.press_event.figure == self
                and event.press_event.button == 1
                and 'Control' in event.press_event.modifiers):
            # stop pan
            delta = self._pxl2nrmView(event.pos-event.press_event.pos)
            self.pan = event.press_event.pan + delta

    def on_key_release(self, event):
        if 'Control' in event.modifiers and event.key == '0':
            self.zoom = [1, 1]
            self.pan  = [0, 0]

    def isInside(self, posPxl):
        return ((self.posPxl < posPxl) &
            (pxl < self.posPxl + self.sizePxl)).all()


class Scope(Figure, pipeline.Sampled):
    def __init__(self, parent, **kwargs):
        '''
        '''

        # properties with their default values
        defaults = dict(tsRange=10, fadeRange=.4)

        self._initProps(defaults, kwargs)

        super().__init__(parent, **kwargs)

    def __setattr__(self, name, value):
        if name in {'tsRange', 'fadeRange'}:
            raise AttributeError('Cannot set attribute')

        super().__setattr__(name, value)

    def initializeGL(self):
        self._tsOffset = 0
        self._tickCount = 10

        self._ticks = [None]*(self._tickCount+1)
        for i, ts in enumerate(np.arange(0, self.tsRange*1.01,
                self.tsRange/self._tickCount)):
            self._ticks[i] = Text(parent=self.view,
                pos=(ts/self.tsRange, 1), anchor=(.5,0), margin=(0,0,0,5),
                fontSize=10, fgColor=self.fgColor, bold=True)

        self.refresh()

        super().initializeGL()

    def refresh(self):
        for i, ts in enumerate(np.arange(0, self.tsRange*1.01,
                self.tsRange/self._tickCount)):
            ts2 = ts + self._tsOffset
            self._ticks[i].text = '%02d:%02d' % (ts2 // 60, ts2 % 60)

    def paintGL(self):
        tsOffset = (self.ts // self.tsRange) * self.tsRange
        if tsOffset != self._tsOffset:
            self._tsOffset = tsOffset
            self.refresh()

        super().paintGL()


class Canvas(Item, QtWidgets.QOpenGLWidget):
    @property
    def canvas(self):
        return self

    def __init__(self, parent, **kwargs):
        # properties with their default values
        defaults = dict(bgColor=(1,1,1,1))

        self._initProps(defaults, kwargs)

        super().__init__(parent, **kwargs)

        self.setMinimumSize(640, 480)

        self._stats = Text(parent=self, text='0', pos=(0,1), anchor=(0,1),
            fgColor=self.fgColor, fontSize=8, bold=True, margin=(2,)*4,
            align=QtCore.Qt.AlignLeft)

        self._drawTimes     = []    # keep last draw times for measuring FPS
        self._drawDurations = []
        self._startTime     = dt.datetime.now()

        self._timerDraw = QtCore.QTimer()
        self._timerDraw.timeout.connect(self.update)
        self._timerDraw.setInterval(1000/60)

        self._timerStats = QtCore.QTimer()
        self._timerStats.timeout.connect(self.updateStats)
        self._timerStats.setInterval(100)

    def updateStats(self):
        drawTime = np.mean(self._drawTimes)
        fps = 1/drawTime if drawTime else 0
        self._fps = fps

        drawDuration = np.mean(self._drawDurations)

        stats = '%.1f Hz\n%.2f ms' % (fps, drawDuration*1e3)
        if hasattr(self, 'stats'): stats += '\n' + self.stats()

        self.makeCurrent()
        self._stats.text = stats

    def initializeGL(self):
        glEnable(GL_BLEND)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
            GL_ONE, GL_ZERO);
        # glClearDepth(1.0)
        # glDepthFunc(GL_LESS)
        # glEnable(GL_DEPTH_TEST)
        # glEnable(GL_MULTISAMPLE)
        # glShadeModel(GL_SMOOTH)

        print('OpenGL Version: %d.%d' % glVersion)
        print('Device Pixel Ratio:',
            QtWidgets.QApplication.screens()[0].devicePixelRatio())
        print('MSAA: x%d' % glSamples)
        print('Max Texture Size: %dx%d' % (
            glGetIntegerv(GL_MAX_RECTANGLE_TEXTURE_SIZE),
            glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE)
            ))

        super().initializeGL()

        self._timerDraw.start()
        self._timerStats.start()

    def resizeGL(self, w, h):
        self._props['posPxl'] = np.array([0, 0])
        self._props['sizePxl'] = np.array([w, h])
        # do not call super().resizeGL() in Canvas
        self.callItems('resizeGL')

    def paintGL(self):
        if hasattr(self, '_drawLast'):
            drawTime = (dt.datetime.now() - self._drawLast).total_seconds()
            self._drawTimes.append(drawTime)
            if len(self._drawTimes)>60: self._drawTimes.pop(0)
        self._drawLast = dt.datetime.now()

        glClearColor(*self.bgColor)
        glClear(GL_COLOR_BUFFER_BIT) # | GL_DEPTH_BUFFER_BIT)

        super().paintGL()

        glFlush()

        drawDuration = (dt.datetime.now() - self._drawLast).total_seconds()
        self._drawDurations.append(drawDuration)
        if len(self._drawDurations)>180: self._drawDurations.pop(0)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.fs = 31.25e3
        self.tsRange = 10
        self.channels = 32

        div = 1
        while (self.channels / div) % 1 != 0 or self.channels / div > 16:
            div += 1
        yTicks = self.channels / div

        self.canvas = Canvas(self)
        self.canvas.stats = lambda: '%.1f s' % (self.physiologyPlot.ts)

        self.scope = Scope(self.canvas, pos=(0,0), size=(.7,1),
            margin=(60,20,20,10), tsRange=self.tsRange,
            xTicks=np.arange(1, self.tsRange)/self.tsRange,
            yTicks=np.r_[np.arange(0, yTicks)/yTicks*.9 + .1, .05])
        self.physiologyPlot = AnalogPlot(self.scope, pos=(0,.1), size=(1,.9))
        self.targetPlot = EpochPlot(self.scope, pos=(0,.05), size=(1,.05),
            name='Target', fgColor=(0,1,0,.6))
        self.pumpPlot = EpochPlot(self.scope, pos=(0,0), size=(1,.05),
            name='Pump', fgColor=(0,0,1,.6))
        self.spikeOverlay = SpikeOverlay(self.scope, pos=(0,.1), size=(1,.9),
            fgColor=(0,0,1,1))

        self.spikeCont = Item(self.canvas, pos=(.7,0), size=(.3,1),
            margin=(0,20,10,10))

        self.spikeFigures = [None] * self.channels
        self.spikePlots   = [None] * self.channels
        div = np.ceil(np.sqrt(self.channels))
        for i in range(self.channels):
            self.spikeFigures[i] = Figure(self.spikeCont,
                pos=(1/div*(i%div), 1-1/div*(i//div+1)), size=(1/div, 1/div),
                margin=(0, 0, 0 if i%div==div-1 else -1,
                    0 if i//div==div-1 else -1),
                xTicks=[.5], yTicks=[.25,.5,.75])
            self.spikePlots[i] = SpikePlot(self.spikeFigures[i],
                name=str(i+1), fgColor=defaultColors[i % len(defaultColors)])

        self.generator = pipeline.SpikeGenerator(fs=self.fs,
            channels=self.channels)
        self.filter    = pipeline.LFilter(fl=100, fh=6000)
        self.grandAvg  = pipeline.GrandAverage()
        self.scale     = 1
        self.scaler    = pipeline.Func(lambda x: x * self.scale)

        self.generator >> self.grandAvg >> self.filter >> (
            self.scope,
            self.scaler >> self.physiologyPlot,
            self.targetPlot.aux,
            self.pumpPlot.aux,
            pipeline.SpikeDetector() >> pipeline.Split() >> self.spikePlots
            )

        # self.generator = pipeline.SineGenerator(fs=self.fs,
        #     channels=self.channels, noisy=True)
        # self.generator >> self.scope >> self.physiologyPlot

        mainLayout = QtWidgets.QHBoxLayout()
        mainLayout.setContentsMargins(0, 0, 0, 0)
        mainLayout.addWidget(self.canvas)

        self.setLayout(mainLayout)
        self.setWindowTitle('OpenGL Demo')
        self.setWindowIcon(QtGui.QIcon(config.APP_LOGO))

    def showEvent(self, event):
        super().showEvent(event)

        if not hasattr(self, '_shown'):
            self._shown = True
            self.generator.start()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            if self.generator.paused: self.generator.start()
            else: self.generator.pause()
        elif event.key() == QtCore.Qt.Key_T:
            self.targetPlot.write(self.generator.ts)
        elif event.key() == QtCore.Qt.Key_P:
            self.pumpPlot.write(self.generator.ts)
        elif event.key() == QtCore.Qt.Key_Up:
            self.scale *= 1.2
        elif event.key() == QtCore.Qt.Key_Down:
            self.scale *= 1/1.2


if __name__ == '__main__':
    def masterExceptionHook(exctype, value, traceback):
        if exctype in (SystemExit, KeyboardInterrupt):
            log.info('Exit requested with "%s"' % exctype.__name__)
        else:
            log.exception('Uncaught exception occured',
                exc_info=(exctype, value, traceback))
        log.info('Exiting application')
        sys.exit(1)
    sys._excepthook = sys.excepthook
    sys.excepthook  = masterExceptionHook

    # setting surface format before running Qt application is mandotary in macOS
    surfaceFormat = QtGui.QSurfaceFormat()
    surfaceFormat.setSamples(glSamples)
    surfaceFormat.setProfile(QtGui.QSurfaceFormat.CoreProfile)
    surfaceFormat.setMajorVersion(glVersion[0])
    surfaceFormat.setMinorVersion(glVersion[1])
    QtGui.QSurfaceFormat.setDefaultFormat(surfaceFormat)

    app = QtWidgets.QApplication([])
    app.setApplicationName = config.APP_NAME
    app.setWindowIcon(QtGui.QIcon(config.APP_LOGO))

    window = MainWindow()
    window.show()

    app.exec_()
