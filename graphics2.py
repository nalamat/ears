'''GPU accelerated graphics and plotting using PyOpenGL.


This file is part of the EARS project: https://github.com/nalamat/ears
Copyright (C) 2017-2020 Nima Alamatsaz <nima.alamatsaz@gmail.com>
Copyright (C) 2017-2020 NESH Lab <ears.software@gmail.com>
Distributed under GNU GPLv3. See LICENSE.txt for more info.
'''

# note: GL color range is 0-1, Qt color range is 0-255

import sys
import math
import time
import logging
import functools
import threading
import numpy      as     np
import scipy      as     sp
import datetime   as     dt
from   scipy      import signal
from   PyQt5      import QtCore, QtGui, QtWidgets
from   ctypes     import sizeof, c_void_p, c_bool, c_uint, c_float, string_at
from   OpenGL.GL  import *

import misc
import config
import pipeline


log = logging.getLogger(__name__)
sizeFloat = sizeof(c_float)

glVersion = (4, 1)

defaultColors = np.array([
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

class Program:
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

    def use(self):
        glUseProgram(self.id)
        glBindVertexArray(self.vao)

    # come up with a better function name
    def unuse(self):
        glBindVertexArray(0)
        glUseProgram(0)

    def setUniform(self, type, name, value):
        glUseProgram(self.id)
        id = glGetUniformLocation(self.id, name)
        globals()['glUniform' + type](id, *value)


class Text():
    _vertShader = '''
        layout (location = 0) in vec2 aVertex;
        layout (location = 1) in vec2 aTexCoord;

        out vec2 TexCoord;

        uniform vec2 uPos;     // unit
        uniform vec2 uAnchor;  // unit
        uniform vec2 uSize;    // pixels
        uniform vec2 uWinSize; // pixels

        void main() {
            gl_Position =
                vec4((aVertex.xy - uAnchor) * 2 * uSize / uWinSize +
                uPos * 2 - vec2(1, 1), 0, 1);
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
        fgColor=(0,0,0,1), bgColor=(0,0,0,0), visible=True, test=False)

    # properties linked with a uniform in the shader program ('format', 'name')
    _uniforms = dict(pos=('2f', 'uPos'), anchor=('2f', 'uAnchor'))

    def __init__(self, **kwargs):
        '''
        Args:
            text (str)
            pos (2 floats): In unit container space.
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
        # initialize properties with the given or default values
        for name, default in self._properties.items():
            if name in kwargs:
                super().__setattr__('_' + name, kwargs[name])
            else:
                super().__setattr__('_' + name, default)

        self._vertices = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
            ], dtype=np.float32)

        self._indices = np.array([
            [0,1,3],
            [1,2,3],
            ], dtype=np.uint32)

        self._texCoords = np.array([
            [0, 1],
            [0, 0],
            [1, 0],
            [1, 1],
            ], dtype=np.float32)

        self._vbo0 = glGenBuffers(1)
        self._vbo1 = glGenBuffers(1)
        self._vbo2 = glGenBuffers(1)
        self._ebo = glGenBuffers(1)
        self._tex = glGenTextures(1)
        self._prog = Program(vert=self._vertShader, frag=self._fragShader)

        self._prog.use()
        self._prog.setUniform('2f', 'uPos', self.pos)
        self._prog.setUniform('2f', 'uAnchor', self.anchor)
        self._prog.setUniform('2f', 'uSize', (0, 0)) # set in update
        self._prog.setUniform('2f', 'uWinSize', (0, 0)) # set in resizeGL

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo0)
        glBufferData(GL_ARRAY_BUFFER, self._vertices, GL_STATIC_DRAW)
        loc = glGetAttribLocation(self._prog.id, 'aVertex')
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(loc)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo1)
        glBufferData(GL_ARRAY_BUFFER, self._texCoords, GL_STATIC_DRAW)
        loc = glGetAttribLocation(self._prog.id, 'aTexCoord')
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(loc)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self._indices, GL_STATIC_DRAW)

        self.update()

        self._prog.unuse()

    def __getattr__(self, name):
        if name in self._properties:
            return super().__getattribute__('_' + name)
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self._properties:
            super().__setattr__('_' + name, value)
            if name in self._uniforms:
                self._prog.setUniform(*self._uniforms[name], value)
            else:
                self.update()
        else:
            super().__setattr__(name, value)

    def update(self):
        '''Updates texture for the given text, font, color, etc.
        Note: This function does not force redrawing of the text on screen.
        '''
        # tic = dt.datetime.now()

        ratio = QtWidgets.QApplication.screens()[0].devicePixelRatio()
        margin = (np.array(self.margin) * ratio).astype(np.int)
        font = QtGui.QFont('arial', self.fontSize * ratio,
            QtGui.QFont.Bold if self.bold else QtGui.QFont.Normal, self.italic)

        w, h = Text.getSize(self.text, font)
        w += margin[0] + margin[2]
        h += margin[1] + margin[3]

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
        self._texData = np.frombuffer(s, dtype=np.uint8).reshape((h, w, 4))

        # toc = (dt.datetime.now() - tic).total_seconds()

        self._prog.setUniform('2f', 'uSize', (w/ratio, h/ratio))

        # texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self._texData.shape[1],
            self._texData.shape[0], 0, GL_BGRA, GL_UNSIGNED_BYTE,
            self._texData)
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glGenerateMipmap(GL_TEXTURE_2D)

        # return toc

    def resizeGL(self, w, h):
        self._prog.setUniform('2f', 'uWinSize', (w, h))

    def draw(self):
        if not self.visible: return

        self._prog.use()

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex)

        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, c_void_p(0))

        self._prog.unuse()


class Container:
    @property
    def canvas(self):
        if self._parent is None:
            raise RuntimeError('No `parent`')
        else:
            return self._parent.canvas

    @property
    def bgColor(self):
        if self._bgColor is not None:
            return self._bgColor
        elif self._parent is None:
            raise RuntimeError('No `parent`')
        else:
            return self._parent.bgColor

    @property
    def fgColor(self):
        if self._fgColor is not None:
            return self._fgColor
        elif self._parent is None:
            raise RuntimeError('No `parent`')
        else:
            return self._parent.fgColor

    def __init__(self, parent=None, untPos=(0,0), untSize=(1,1),
            pxlMargin=(0,0,0,0), bgColor=None, fgColor=None, **kwargs):
        '''
        Args:
            untPos (2 floats): Container position in unit parent space: (x, y).
            untSize (2 floats): Container size in unit parent space: (w, h).
            pxlMargin (4 floats): Margin in pixels: (l, t, r, b).
            bgColor (3 floats): Backgoround color. If None,
                defaults to parent.bgColor.
            fgColor (3 floats): Foreground color for border and texts. If None,
                defaults to parent.fgColor.
        '''

        # allow multiple inheritance and mixing with other classes
        super().__init__(**kwargs)

        self._parent    = parent
        self._untPos    = np.array(untPos)
        self._untSize   = np.array(untSize)
        self._pxlMargin = np.array(pxlMargin)
        self._pxlPos    = np.array([0, 0])
        self._pxlSize   = np.array([1, 1])
        self._bgColor   = bgColor
        self._fgColor   = fgColor
        self._items     = []

        if parent is not None:
            parent.addItem(self)

    def _callItems(self, func, *args, **kwargs):
        for item in self._items:
            if hasattr(item, func):
                getattr(item, func)(*args, **kwargs)

    def addItem(self, item):
        # if not isinstance(item, Container):
        #     raise TypeError('`item` should be a Container')

        self._items += [item]

    def on_resize(self, *args, **kwargs):
        parent = self._parent

        self._pxlPos = (parent._pxlPos + self._untPos * (parent._pxlSize
            - parent._pxlMargin[0:2] - parent._pxlMargin[2:4])
            + parent._pxlMargin[0:2])
        self._pxlSize = self._untSize * (parent._pxlSize
            - parent._pxlMargin[0:2] - parent._pxlMargin[2:4])

        self._callItems('on_resize', *args, **kwargs)

    _funcs = ['on_mouse_wheel', 'on_mouse_press', 'on_mouse_move',
        'on_mouse_release', 'on_key_release', 'draw']
    for func in _funcs:
        exec('def %(func)s(self, *args, **kwargs):\n'
            '    self._callItems("%(func)s", *args, **kwargs)' % {'func':func})


class Canvas(QtWidgets.QOpenGLWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self._bg = (1, 1, 1, 1)

        self.setMinimumSize(640, 480)

        self._items = []

        self._updateTimes = []    # keep last update times for measuring FPS
        self._startTime   = dt.datetime.now()

        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self.update)
        self._timer.setInterval(1000/60)
        self._timer.start()

    def initializeGL(self):
        self._text1 = Text(text='Hello GL', pos=(.5,.5), fontSize=100,
            fgColor=(0,0,.3,1))
        self._fpsText = Text(text='0', pos=(0, 1), anchor=(0, 1),
            fgColor=(0,0,.3,1), margin=(2,2,2,2), fontSize=8, bold=True)
        self._items += [self._text1, self._fpsText]

        # glClearDepth(1.0)
        # glDepthFunc(GL_LESS)
        # glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glShadeModel(GL_SMOOTH)
        glClearColor(*self._bg)

    def resizeGL(self, w, h):
        for item in self._items:
            item.resizeGL(w, h)

    def paintGL(self):
        if hasattr(self, '_updateLast'):
            updateTime = (dt.datetime.now() -
                self._updateLast).total_seconds()
            self._updateTimes.append(updateTime)
            if len(self._updateTimes)>60:
                self._updateTimes.pop(0)
            updateMean = np.mean(self._updateTimes)
            fps = 1/updateMean if updateMean else 0
            self._fps = fps
            self._fpsText.text = '%.1f' % fps
        self._updateLast = dt.datetime.now()

        glClear(GL_COLOR_BUFFER_BIT) # | GL_DEPTH_BUFFER_BIT)

        for item in self._items:
            item.draw()

        glFlush()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('OpenGL Demo')
        self.setWindowIcon(QtGui.QIcon(config.APP_LOGO))

        self.canvas = Canvas(self)

        mainLayout = QtWidgets.QHBoxLayout()
        mainLayout.setContentsMargins(0, 0, 0, 0)
        mainLayout.addWidget(self.canvas)

        self.setLayout(mainLayout)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            print('Space pressed')


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
