'''GPU accelerated graphics and plotting using PyOpenGL.


This file is part of the EARS project: https://github.com/nalamat/ears
Copyright (C) 2017-2020 Nima Alamatsaz <nima.alamatsaz@gmail.com>
Copyright (C) 2017-2020 NESH Lab <ears.software@gmail.com>
Distributed under GNU GPLv3. See LICENSE.txt for more info.
'''

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

glslTransform = '''
    vec2 transform(vec2 p, vec2 a, vec2 b) {
        return a*p + b;
    }

    vec2 transform(vec2 p, float ax, float ay, float bx, float by) {
        return transform(p, vec2(ax, ay), vec2(bx, by));
    }
    '''

glslRand = '''
    float rand(vec2 seed){
        return fract(sin(dot(seed.xy ,vec2(12.9898,78.233))) * 43758.5453);
    }
    '''

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
    @staticmethod
    def createShader(shaderType, source):
        """Compile a shader."""
        source = ('#version %d%d0 core\n' % glVersion) + source
        shader = glCreateShader(shaderType)
        glShaderSource(shader, source)
        glCompileShader(shader)

        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(shader))

        return shader

    def __init__(self, vert=None, geom=None, frag=None, comp=None):
        if vert: vertShader = Program.createShader(GL_VERTEX_SHADER  , vert)
        if geom: geomShader = Program.createShader(GL_GEOMETRY_SHADER, geom)
        if frag: fragShader = Program.createShader(GL_FRAGMENT_SHADER, frag)
        if comp: compShader = Program.createShader(GL_COMPUTE_SHADER , comp)

        self.id = glCreateProgram()

        if vert: glAttachShader(self.id, vertShader)
        if geom: glAttachShader(self.id, geomShader)
        if frag: glAttachShader(self.id, fragShader)
        if comp: glAttachShader(self.id, compShader)

        glLinkProgram(self.id)

        if glGetProgramiv(self.id, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(self.id))

        if vert: glDeleteShader(vertShader)
        if geom: glDeleteShader(geomShader)
        if frag: glDeleteShader(fragShader)
        if comp: glDeleteShader(compShader)

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
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoord;

        out vec2 TexCoord;

        uniform vec2 uPos;
        uniform vec2 uAnchor;
        uniform vec2 uSize;
        uniform vec2 uWinSize;

        void main() {
            gl_Position =
                vec4((aPos.xy - uAnchor) * 2 * uSize / uWinSize +
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
        '''Get width and height of the given text'''
        if not hasattr(cls, '_image'):
            cls._image = QtGui.QPixmap(1, 1)
            cls._painter = QtGui.QPainter()
            cls._painter.begin(cls._image)

        cls._painter.setFont(font)
        rect = cls._painter.boundingRect(QtCore.QRect(),
            QtCore.Qt.AlignLeft, text)

        return rect.width(), rect.height()

    # define updatable properties
    _props = ['text', 'fontSize', 'fgColor', 'bgColor', 'align']
    for prop in _props:
        exec(
            '@property\n'
            'def %(prop)s(self):\n'
            '    return self._%(prop)s\n'
            '@%(prop)s.setter\n'
            'def %(prop)s(self, value):\n'
            '    self._%(prop)s = value\n'
            '    self.update()\n'
            % {'prop': prop}
            )

    def __init__(self, text, pos, fontSize=12, fgColor=(255,255,255,255),
            bgColor=(0,0,0,0), align=QtCore.Qt.AlignCenter,
            anchor=(.5,.5), visible=True, test=False):

        self._text = text
        self._pos = pos
        self._fontSize = fontSize
        self._fgColor = fgColor
        self._bgColor = bgColor
        self._align = align

        self.anchor = anchor
        self.visible = visible
        self.test = test

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
        self._prog.setUniform('2f', 'uPos', pos)
        self._prog.setUniform('2f', 'uAnchor', anchor)
        self._prog.setUniform('2f', 'uSize', (0, 0)) # set in update
        self._prog.setUniform('2f', 'uWinSize', (0, 0)) # set in resizeGL

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo0)
        glBufferData(GL_ARRAY_BUFFER, self._vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo1)
        glBufferData(GL_ARRAY_BUFFER, self._texCoords, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self._indices, GL_STATIC_DRAW)

        self.update()

        self._prog.unuse()

    def update(self):
        '''Updates texture for the given text, font, color, etc
        Note: This function does not force redrawing of the text on screen
        '''
        tic = dt.datetime.now()

        ratio = QtWidgets.QApplication.screens()[0].devicePixelRatio()
        font = QtGui.QFont('arial', self._fontSize * ratio)
        w, h = Text.getSize(self._text, font)

        image = QtGui.QPixmap(w, h)
        image.fill(QtGui.QColor(*self._bgColor))

        painter = QtGui.QPainter()
        painter.begin(image)
        painter.setPen(QtGui.QColor(*self._fgColor))
        painter.setFont(font)
        painter.drawText(0, 0, w, h, self._align, self._text)
        painter.end()
        image = image.toImage()
        s = image.bits().asstring(w * h * 4)
        self._texData = np.frombuffer(s, dtype=np.uint8).reshape((h, w, 4))

        toc = (dt.datetime.now() - tic).total_seconds()

        self._prog.setUniform('2f', 'uSize', (w/ratio, h/ratio))

        # texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self._texData.shape[1],
            self._texData.shape[0], 0, GL_BGRA, GL_UNSIGNED_BYTE,
            self._texData)
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glGenerateMipmap(GL_TEXTURE_2D)

        return toc

    def resizeGL(self, w, h):
        self._prog.setUniform('2f', 'uWinSize', (w, h))

    def draw(self):
        if not self.visible: return

        self._prog.use()

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex)

        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, c_void_p(0))

        self._prog.unuse()


class Canvas(QtWidgets.QOpenGLWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self._bg = np.array([0, 0, 0.2])

        self.setMinimumSize(640, 480)

        self._items = []

        self._updateLast  = None
        self._updateTimes = []    # keep last update times for measuring FPS
        self._startTime   = dt.datetime.now()

        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self.update)
        self._timer.setInterval(1000/60)
        self._timer.start()

    def initializeGL(self):
        for i in range(600):
            t = Text(str(np.round(np.random.rand()*1000, 3)), np.random.rand(2),
                int(np.random.rand()*90+10), anchor=np.random.rand(2),
                fgColor=np.floor(np.random.rand(4)*256))
            self._items += [t]

        # self._text1 = Text('Hello', (.25, .5), 100)
        # self._text2 = Text('Hello', (.75, .5), 100, test=True)
        self._fpsText = Text('0', (0, 1), 12, anchor=(0, 1))
        self._items += [self._fpsText]

        # glClearDepth(1.0)
        # glDepthFunc(GL_LESS)
        # glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glShadeModel(GL_SMOOTH)
        glClearColor(*self._bg, 0)

    def resizeGL(self, w, h):
        for item in self._items:
            item.resizeGL(w, h)

    def paintGL(self):
        if self._updateLast is not None:
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
