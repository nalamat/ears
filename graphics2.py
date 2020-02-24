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
#   however, OpenGL coordinates in fragment shader are actual pixel values and
#   must be normalized manually if necessary
# -

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

defaultColors = np.array([
    [0 , 0,  .3, 1],    # 1
    [0 , 0 , .6, 1],    # 2
    [0 , 0 , 1 , 1],    # 3
    [0 , .3, 1 , 1],    # 4
    [0 , .6, 1 , 1],    # 5
    [0 , .6, .6, 1],    # 6
    [0 , .3, 0 , 1],    # 7
    [0 , .6, 0 , 1],    # 8
    [0 , .9, 0 , 1],    # 9
    [.5, .6, 0 , 1],    # 10
    [1 , .6, 0 , 1],    # 11
    [.3, 0 , 0 , 1],    # 12
    [.6, 0 , 0 , 1],    # 13
    [1 , 0 , 0 , 1],    # 14
    [1 , 0 , .3, 1],    # 15
    [1 , 0 , .6, 1],    # 16
    [.6, 0 , .6, 1],    # 17
    [.6, 0 , 1 , 1],    # 18
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

        float rand(vec2 seed){
            return fract(sin(dot(seed.xy, vec2(12.9898,78.233))) * 43758.5453);
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

    def begin(self):
        glUseProgram(self.id)
        glBindVertexArray(self.vao)

    def end(self):
        glBindVertexArray(0)
        glUseProgram(0)

    def setUniform(self, name, type, value):
        if not isinstance(value, collections.abc.Iterable): value = (value,)
        glUseProgram(self.id)
        id = glGetUniformLocation(self.id, name)
        globals()['glUniform' + type](id, *value)
        glUseProgram(0)

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
        self.begin()
        if name not in self.vbos: self.vbos[name] = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbos[name])
        glBufferData(GL_ARRAY_BUFFER, data, usage)
        loc = glGetAttribLocation(self.id, name)
        glVertexAttribPointer(loc, size, type, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(loc)
        self.end()

    def setEBO(self, data, usage):
        self.begin()
        if not self.ebo: self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, data, usage)
        self.end()


class Item:
    '''Graphical item capable of containing child items.'''
    @property
    def canvas(self):
        if self.parent:
            return self.parent.canvas
        else:
            return None

    # properties with their default values
    _properties = dict(parent=None, pos=(0,0), size=(1,1), margin=(0,0,0,0),
        bgColor=(0,0,0,0), fgColor=(0,0,0,1), visible=True)

    def __init__(self, **kwargs):
        '''
        Args:
            parent
            pos (2 floats): Item position in unit parent space (x, y).
            size (2 floats): Item size in unit parent space (w, h).
            margin (4 floats): Margin in pixels (l, t, r, b).
            bgColor (3 floats): Backgoround color.
            fgColor (3 floats): Foreground color for border and texts.
        '''
        super().__init__()
        self._initProperties(Item._properties, kwargs)

        self._posPxl    = np.array([0, 0])
        self._sizePxl   = np.array([1, 1])
        self._items     = []

        if self.parent:
            self.parent.addItem(self)

    def __getattr__(self, name):
        if name in Item._properties or name in {'posPxl', 'sizePxl'}:
            return super().__getattribute__('_' + name)
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in {'bgColor', 'fgColor'}:
            super().__setattr__('_' + name, value)
        else:
            super().__setattr__(name, value)

    def _initProperties(self, properties, kwargs):
        # initialize properties with the given or default values
        # setter monitoring (updates, etc) won't apply here
        for name, default in properties.items():
            if hasattr(self, '_' + name):
                continue
            elif name in kwargs:
                setattr(self, '_' + name, kwargs[name])
            else:
                setattr(self, '_' + name, default)

    def addItem(self, item):
        # if not isinstance(item, Item):
        #     raise TypeError('`item` should be a Item')

        self._items += [item]

    def callItems(self, func, *args, **kwargs):
        for item in self._items:
            if hasattr(item, func):
                getattr(item, func)(*args, **kwargs)

    # initally called from Canvas.resizeGL
    def parentResized(self):
        parent = self.parent
        margin = np.array(self.margin)

        self._posPxl = self.pos * parent.sizePxl + parent.posPxl + margin[[0,3]]
        self._sizePxl = self.size * parent.sizePxl - margin[0:2] - margin[2:4]

        self.callItems('parentResized')

    _funcs = ['on_mouse_wheel', 'on_mouse_press', 'on_mouse_move',
        'on_mouse_release', 'on_key_release', 'draw']
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

        uniform sampler2D ourTexture;

        void main() {
            FragColor = texture(ourTexture, TexCoord);
        }
        '''

    @classmethod
    def getSize(cls, text, font):
        '''Get width and height of the given text in pixels'''
        if not hasattr(cls, '_image'):
            cls._image = QtGui.QPixmap(1, 1)
            cls._painter = QtGui.QPainter()
            cls._painter.begin(cls._image)

        cls._painter.setFont(font)
        rect = cls._painter.boundingRect(QtCore.QRect(),
            QtCore.Qt.AlignLeft, text)

        return rect.width(), rect.height()

    # properties with their default values
    _properties = dict(text='', pos=(.5,.5), anchor=(.5,.5), margin=(0,0,0,0),
        fontSize=12, bold=False, italic=False, align=QtCore.Qt.AlignCenter,
        bgColor=(0,0,0,0), fgColor=(0,0,0,1))

    def __init__(self, **kwargs):
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

        self._initProperties(Text._properties, kwargs)
        super().__init__(**kwargs)

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

        self._prog.setUniform('uPos', '2f', self.pos)
        self._prog.setUniform('uAnchor', '2f', self.anchor)
        self._prog.setUniform('uSize', '2f', (0,0)) # set in update
        self._prog.setUniform('uCanvasSize', '2f',
            self.canvas.sizePxl if self.canvas else (0,0))

        self._prog.setVBO('aVertex', GL_FLOAT, 2, vertices, GL_STATIC_DRAW)
        self._prog.setVBO('aTexCoord', GL_FLOAT, 2, texCoords, GL_STATIC_DRAW)
        self._prog.setEBO(indices, GL_STATIC_DRAW)

        self.update()

    def __getattr__(self, name):
        if name in Text._properties:
            return super().__getattr__('_' + name)
        else:
            return super().__getattr__(name)

    def __setattr__(self, name, value):
        if name in {'pos', 'anchor'}:
            super().__setattr__('_' + name, value)
            self._prog.setUniform('u'+name[0].upper()+name[1:], '2f', value)
        elif name in Text._properties:
            super().__setattr__('_' + name, value)
            self.update()
        else:
            super().__setattr__(name, value)

    def parentResized(self):
        self._prog.setUniform('uParentPos', '2f', self.parent.posPxl)
        self._prog.setUniform('uParentSize', '2f', self.parent.sizePxl)
        self._prog.setUniform('uCanvasSize', '2f', self.canvas.sizePxl)

        super().parentResized()

    def update(self):
        '''Updates texture for the given text, font, color, etc.
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

        image = QtGui.QPixmap(w, h)
        image.fill(QtGui.QColor(*np.array(self.bgColor)*255))

        painter = QtGui.QPainter()
        painter.begin(image)
        painter.setPen(QtGui.QColor(*np.array(self.fgColor)*255))
        painter.setFont(font)
        painter.drawText(margin[0], margin[1],
            w - margin[0] - margin[2], h - margin[1] - margin[3],
            self.align, self.text)
        painter.end()
        image = image.toImage()
        s = image.bits().asstring(w * h * 4)
        texData = np.frombuffer(s, dtype=np.uint8).reshape((h, w, 4))

        self._prog.setUniform('uSize', '2f', (w/ratio, h/ratio))

        # texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texData.shape[1],
            texData.shape[0], 0, GL_BGRA, GL_UNSIGNED_BYTE, texData)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glGenerateMipmap(GL_TEXTURE_2D)

    def draw(self):
        if not self.visible: return

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex)

        self._prog.begin()
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, c_void_p(0))
        self._prog.end()

        super().draw()


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
            gl_Position = vec4( (aVertex * size + pos) * 2 - vec2(1), 0, 1);
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

        bool between(float a, float b, float c) {
            return a <= b && b < c;
        }

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

    _uniforms = dict(pos=('uPos', '2f'), size=('uSize', '2f'),
        margin=('uMargin', '4f'), borderWidth=('uBorderWidth', '1f'),
        bgColor=('uBgColor', '4f'), fgColor=('uFgColor', '4f'))

    _properties = dict(borderWidth=1)

    def __init__(self, **kwargs):
        '''
        '''

        self._initProperties(Rectangle._properties, kwargs)
        super().__init__(**kwargs)

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

        self._prog = Program(vert=self._vertShader,
                             frag=self._fragShader)

        for name, value in Rectangle._uniforms.items():
            self._prog.setUniform(*value, getattr(self, name))
        # high DPI display support
        self._prog.setUniform('uPixelRatio', '1f', getPixelRatio())

        self._prog.setVBO('aVertex', GL_FLOAT, 2, vertices, GL_STATIC_DRAW)
        self._prog.setEBO(self._indices, GL_STATIC_DRAW)

    def __getattr__(self, name):
        if name in Rectangle._properties:
            return super().__getattr__('_' + name)
        else:
            return super().__getattr__(name)

    def __setattr__(self, name, value):
        if name in Rectangle._uniforms:
            super().__setattr__('_' + name, value)
            self._prog.setUniform(*Rectangle._uniforms[name], value)
        else:
            super().__setattr__(name, value)

    def parentResized(self):
        self._prog.setUniform('uParentPos', '2f', self.parent.posPxl)
        self._prog.setUniform('uParentSize', '2f', self.parent.sizePxl)
        self._prog.setUniform('uCanvasSize', '2f', self.canvas.sizePxl)

        super().parentResized()

    def draw(self):
        if not self.visible: return

        self._prog.begin()
        glDrawElements(GL_TRIANGLES, self._indices.size,
            GL_UNSIGNED_INT, c_void_p(0))
        self._prog.end()

        super().draw()


class Figure(Item):
    @property
    def zoom(self):
        return self._zoom

    @zoom.setter
    def zoom(self, zoom):
        zoom = np.array(zoom)
        zoom = zoom.clip(1, 1000)
        self._zoom = zoom
        self._transformView['zoom'] = zoom
        self.callItems('zoomed')
        # self.canvas.update()

    @property
    def pan(self):
        return self._pan

    @pan.setter
    def pan(self, pan):
        # `pan` is normalized in view space
        panLimits = self._zoom-1
        pan = np.array(pan)
        pan = pan.clip(-panLimits, panLimits)
        self._pan = pan
        self._transformView['pan'] = pan
        self.callItems('panned')
        # self.canvas.update()

    _properties = dict(borderWidth=2, zoom=(1,1),
        pan=(0,0))

    def __init__(self, **kwargs):
        '''
        '''

        self._initProperties(Figure._properties, kwargs)
        super().__init__(**kwargs)


        # setup glsl transformations for child plot items
        # self._transformView = vp.visuals.shaders.Function('''
        #     vec2 transform_view(vec2 pos) {
        #         return $transform(pos, $zoom, $pan);
        #     }''')
        # self._transformView['transform'] = _glslTransform
        # self._transformView['zoom'     ] = self._zoom
        # self._transformView['pan'      ] = self._pan

        # self._transformFigure = vp.visuals.shaders.Function('''
        #     vec2 transform_figure(vec2 pos) {
        #         return $transform(pos,
        #             $view_size / $figure_size,
        #             ($margins.xy - $margins.wz) / $figure_size
        #             * vec2(1,-1));
        #     }''')
        # self._transformFigure['transform'] = _glslTransform
        # self._transformFigure['margins'  ] = self._pxlMargins

        # self._transformCanvas = vp.visuals.shaders.Function('''
        #     vec2 transform_canvas(vec2 pos) {
        #         return $transform(pos,
        #             $figure_size / $canvas_size,
        #             (($figure_pos + $figure_size/2) * 2 / $canvas_size - 1)
        #             * vec2(1,-1));
        #     }''')
        # self._transformCanvas['transform'] = _glslTransform

        # a border around the figure
        self._view = Rectangle(parent=self, pos=(0,0), size=(1,1),
            borderWidth=self.borderWidth, bgColor=self.bgColor,
            fgColor=self.fgColor)

        self._plot = AnalogPlot(parent=self._view, margin=(self.borderWidth,)*4)

    def __getattr__(self, name):
        if name in Figure._properties:
            return super().__getattr__('_' + name)
        else:
            return super().__getattr__(name)

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

    def isInside(self, pxl):
        return ((self._pxlPos < pxl) & (pxl < self._pxlPos+self._pxlSize)).all()


class AnalogPlot(Item):
    _vertShader = '''
        in float aVertex;

        out float fChannel;

        uniform vec2  uPos;
        uniform vec2  uSize;
        uniform vec4  uMargin;
        uniform vec2  uParentPos;
        uniform vec2  uParentSize;
        uniform vec2  uCanvasSize;
        uniform float uTsRange;
        uniform float uChannels;
        uniform float uFs;

        void main() {
            float count = int(uTsRange * uFs);
            float channel = int(gl_VertexID / count);
            vec2 vertex = vec2((gl_VertexID % int(count)) / (count - 1),
                ((aVertex + 1) / 2 + uChannels - 1 - channel) / uChannels);

            vec2 pos = (uPos * uParentSize + uParentPos) / uCanvasSize;
            vec2 size = uSize * uParentSize / uCanvasSize;

            // add margin
            pos += uMargin.xw / uCanvasSize;
            size -= (uMargin.xy + uMargin.zw) / uCanvasSize;

            vertex = (vertex * size + pos) * 2 - vec2(1);

            gl_Position = vec4(vertex, 0, 1);
            fChannel = channel;
        }
        '''

    _fragShader = '''
        #define MAX_CHANNELS 128

        in float fChannel;
        out vec4 FragColor;

        uniform vec4  uColor[MAX_CHANNELS];

        void main() {
            if (0 < fract(fChannel)) discard;
            FragColor = uColor[int(fChannel)];
        }
        '''

    _uniforms = dict(pos=('uPos', '2f'), size=('uSize', '2f'),
        margin=('uMargin', '4f'))

    _properties = dict()

    def __init__(self, **kwargs):
        self._initProperties(AnalogPlot._properties, kwargs)
        super().__init__(**kwargs)


        self._tsRange = 20
        self._channels = 16
        self._fs = 31.25e3

        indices = np.array([np.arange(self._fs*self._tsRange)]*self._channels)
        self._vertices = np.sin(indices/self._fs*2*np.pi).astype(np.float32)

        self._prog = Program(vert=self._vertShader,
                             frag=self._fragShader)

        for name, value in AnalogPlot._uniforms.items():
            self._prog.setUniform(*value, getattr(self, name))
        self._prog.setUniform('uTsRange', '1f', self._tsRange)
        self._prog.setUniform('uChannels', '1f', self._channels)
        self._prog.setUniform('uFs', '1f', self._fs)
        self._prog.setUniform('uColor', '4fv', (self._channels, defaultColors))

        self._prog.setVBO('aVertex', GL_FLOAT, 1, self._vertices,
            GL_DYNAMIC_DRAW)

        self._labels = [None] * self._channels
        for channel in range(self._channels):
            self._labels[channel] = Text(parent=self, text=str(channel+1),
                pos=(0, 1-(channel+.5)/self._channels), anchor=(1,.5),
                margin=(0,0,3+self.margin[0],0),
                fontSize=16, bold=True, fgColor=defaultColors[channel])

        self._ticks = [None]*11
        for i, ts in enumerate(np.arange(0, self._tsRange*1.01,
                self._tsRange/10)):
            self._ticks[i] = Text(parent=self, text=str(ts),
                pos=(ts/self._tsRange, 1), anchor=(.5,0),
                margin=(0,0,0,3+self.margin[1]),
                fontSize=10, fgColor=self.fgColor)

    def __getattr__(self, name):
        if name in AnalogPlot._properties:
            return super().__getattr__('_' + name)
        else:
            return super().__getattr__(name)

    def __setattr__(self, name, value):
        if name in AnalogPlot._uniforms:
            super().__setattr__('_' + name, value)
            self._prog.setUniform(*AnalogPlot._uniforms[name], value)
        else:
            super().__setattr__(name, value)

    def parentResized(self):
        self._prog.setUniform('uParentPos', '2f', self.parent.posPxl)
        self._prog.setUniform('uParentSize', '2f', self.parent.sizePxl)
        self._prog.setUniform('uCanvasSize', '2f', self.canvas.sizePxl)

        super().parentResized()

    def draw(self):
        if not self.visible: return

        self._prog.begin()
        glDrawArrays(GL_LINE_STRIP, 0, self._vertices.size)
        self._prog.end()

        super().draw()


class Canvas(Item, QtWidgets.QOpenGLWidget):
    @property
    def canvas(self):
        return self

    # `parent` needs to be defined out of __getattr__ to override
    # QOpenGLWidget's property
    @property
    def parent(self):
        return self._parent

    # properties with their default values
    _properties = dict(bgColor=(1,1,1,1))

    def __init__(self, **kwargs):
        self._initProperties(Canvas._properties, kwargs)

        super().__init__(**kwargs)

        self.setMinimumSize(640, 480)

        self._drawTimes     = []    # keep last draw times for measuring FPS
        self._drawDurations = []
        self._startTime     = dt.datetime.now()

        self._timerDraw = QtCore.QTimer()
        self._timerDraw.timeout.connect(self.update)
        self._timerDraw.setInterval(1000/60)
        self._timerDraw.start()

        self._timerStats = QtCore.QTimer()
        self._timerStats.timeout.connect(self.updateStats)
        self._timerStats.setInterval(100)

    def __setattr__(self, name, value):
        if name == 'bgColor':
            super().__setattr__('_' + name, value)
            self.makeCurrent()
            glClearColor(*value)
        else:
            super().__setattr__(name, value)

    def updateStats(self):
        drawTime = np.mean(self._drawTimes)
        fps = 1/drawTime if drawTime else 0
        self._fps = fps

        drawDuration = np.mean(self._drawDurations)
        self._stats.text = '%.1f Hz\n%.1f μs' % (fps, drawDuration*1e6)

    def initializeGL(self):
        # initialize graphical items here
        # self._text = Text(parent=self, text='Hello GL', pos=(.5,1),
        #     anchor=(.5,1), margin=(0,30,0,0), fontSize=50,
        #     fgColor=self.fgColor)

        self._figure = Figure(parent=self, margin=(60,20,20,10))

        item = Item(parent=self, margin=(2,)*4)
        self._stats = Text(parent=item, text='0', pos=(0,1), anchor=(0,1),
            fgColor=self.fgColor, fontSize=8, bold=True, margin=(1,)*4,
            align=QtCore.Qt.AlignLeft)

        # glClearDepth(1.0)
        # glDepthFunc(GL_LESS)
        # glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
        # glShadeModel(GL_SMOOTH)
        glClearColor(*self.bgColor)

        self._timerStats.start()

    def resizeGL(self, w, h):
        self._posPxl = np.array([0, 0])
        self._sizePxl = np.array([w, h])
        self.callItems('parentResized')

    def paintGL(self):
        if hasattr(self, '_drawLast'):
            drawTime = (dt.datetime.now() - self._drawLast).total_seconds()
            self._drawTimes.append(drawTime)
            if len(self._drawTimes)>60: self._drawTimes.pop(0)
        self._drawLast = dt.datetime.now()

        glClear(GL_COLOR_BUFFER_BIT) # | GL_DEPTH_BUFFER_BIT)

        self.callItems('draw')

        glFlush()

        drawDuration = (dt.datetime.now() - self._drawLast).total_seconds()
        self._drawDurations.append(drawDuration)
        if len(self._drawDurations)>180: self._drawDurations.pop(0)


class Generator(pipeline.Sampled):
    @property
    def paused(self):
        return not self._continueEvent.isSet()

    def __init__(self, fs, channels, **kwargs):
        self._ns = 0
        self._thread = threading.Thread(target=self._loop)
        self._thread.daemon = True
        self._continueEvent = threading.Event()

        super().__init__(fs=fs, channels=channels, **kwargs)

    def _addSources(self, sources):
        raise RuntimeError('Cannot add source for a Generator')

    def _loop(self):
        start = dt.datetime.now()
        pauseTS = 0

        while True:
            time.sleep(.01)
            if not self._continueEvent.isSet():
                pause = dt.datetime.now()
                self._continueEvent.wait()
                pauseTS += (dt.datetime.now()-pause).total_seconds()
            ts = (dt.datetime.now()-start).total_seconds()-pauseTS
            ns = int(ts * self._fs)
            data = self._gen(self._ns, ns)
            super().write(data, self)
            self._ns = ns

    def _gen(self, ns1, ns2):
        raise NotImplementedError()

    def start(self):
        self._continueEvent.set()
        if not self._thread.isAlive():
            self._thread.start()

    def pause(self):
        self._continueEvent.clear()

    def write(self, data, source):
        raise RuntimeError('Cannot write to a Generator')


class SineGenerator(Generator):
    def __init__(self, fs, channels, noisy=True, **kwargs):
        # randomized channel parameters
        self._phases = np.random.uniform(0  , np.pi, (channels,1))
        self._freqs  = np.random.uniform(1  , 5    , (channels,1))
        self._amps   = np.random.uniform(.05, .5   , (channels,1))
        self._amps2  = np.random.uniform(.05, .2   , (channels,1))
        if not noisy:
            self._amps2 *= 0

        super().__init__(fs=fs, channels=channels, **kwargs)

    def _gen(self, ns1, ns2):
        # generate data as a (channels, ns) array.
        return (self._amps * np.sin(2 * np.pi * self._freqs
            * np.arange(ns1, ns2) / self._fs + self._phases)
            + self._amps2 * np.random.randn(self._channels, ns2-ns1))


class SpikeGenerator(Generator):
    def _gen(self, ns1, ns2):
        data = .1*np.random.randn(self._channels, ns2-ns1)
        dt = (ns2-ns1)/self._fs

        counts = np.random.uniform(0, dt*1000/30, self._channels)
        counts = np.round(counts).astype(np.int)

        for channel in range(self._channels):
            for i in range(counts[channel]):
                # random spike length, amplitude, and location
                length = int(np.random.uniform(.5e-3, 1e-3)*self._fs)
                amp    = np.random.uniform(.3, 1)
                at     = int(np.random.uniform(0, data.shape[1]-length))
                spike  = -amp*sp.signal.gaussian(length, length/7)
                data[channel, at:at+length] += spike

        return data


class SpikeDetector(pipeline.Sampled):
    def __init__(self, tl=4, th=20, spikeDuration=2e-3, **kwargs):
        '''
        Args:
            tl (float): Lower detection threshold
            th (float): Higher detection threshold
            spikeDuration (float): Spike window duration
        '''
        self._tl            = tl
        self._th            = th
        self._spikeDuration = spikeDuration
        self._spikeLength   = None
        self._buffer        = None
        self._sd            = None

        super().__init__(**kwargs)

    def _configured(self, params, sinkParams):
        self._spikeLength         = int(self._spikeDuration*params['fs'])
        sinkParams['spikeLength'] = self._spikeLength

        super()._configured(params, sinkParams)

        self._buffer = misc.CircularBuffer((self._channels, int(self._fs*10)))

    def _written(self, data, source):
        self._buffer.write(data)

        nsRead      = self._buffer.nsRead
        nsAvailable = self._buffer.nsAvailable

        if (nsRead<self._fs*5 and nsAvailable>self._fs
                or nsRead>self.fs*5 and nsAvailable>self._fs*5):
            self._sd = np.median(np.abs(self._buffer.read()))/0.6745;

        # do not detect spikes until enough samples are available for
        # calculating standard deviation of noise
        if self._sd is None: return

        windowHalf  = self._spikeLength/2
        windowStart = int(np.floor(windowHalf))
        windowStop  = int(np.ceil(windowHalf))
        dataOut     = [None]*self._channels

        # TODO: append last half window samples of previous data to current data
        for i in range(self._channels):
            dataOut[i] = []

            peaks, _ = sp.signal.find_peaks(-data[i],
                height=(self._tl*self._sd, self._th*self._sd),
                distance=windowHalf)
            for peak in peaks:
                if 0 <= peak-windowStart and peak+windowStop < len(data[i]):
                    dataOut[i] += [data[i,peak-windowStart:peak+windowStop]]

        super()._written(dataOut, source)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('OpenGL Demo')
        self.setWindowIcon(QtGui.QIcon(config.APP_LOGO))

        self.canvas = Canvas()

        mainLayout = QtWidgets.QHBoxLayout()
        mainLayout.setContentsMargins(0, 0, 0, 0)
        mainLayout.addWidget(self.canvas)

        self.setLayout(mainLayout)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            # print('Space pressed')
            if self.canvas._text1.fontSize > 150:
                self.canvas._text1._step = -5
            if self.canvas._text1.fontSize < 20 or \
                    not hasattr(self.canvas._text1, '_step'):
                self.canvas._text1._step = +5
            self.canvas._text1.fontSize += self.canvas._text1._step


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
    # surfaceFormat.setSamples(16)
    surfaceFormat.setProfile(QtGui.QSurfaceFormat.CoreProfile)
    surfaceFormat.setMajorVersion(glVersion[0])
    surfaceFormat.setMinorVersion(glVersion[1])
    QtGui.QSurfaceFormat.setDefaultFormat(surfaceFormat)

    app = QtWidgets.QApplication([])
    app.setApplicationName = config.APP_NAME
    app.setWindowIcon(QtGui.QIcon(config.APP_LOGO))

    print('OpenGL Version: %d.%d' % glVersion)
    print('Device Pixel Ratio:', app.screens()[0].devicePixelRatio())

    window = MainWindow()
    window.show()

    app.exec_()
