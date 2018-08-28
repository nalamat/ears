'''GPU accelerated graphics and plotting using vispy.


This file is part of the EARS project: https://github.com/nalamat/ears
Copyright (C) 2017-2018 Nima Alamatsaz <nima.alamatsaz@njit.edu>
Copyright (C) 2017-2018 Antje Ihlefeld <antje.ihlefeld@njit.edu>
Distrubeted under GNU GPLv3. See LICENSE.txt for more info.
'''

import math
import time
import threading
import numpy      as     np
import vispy      as     vp
import datetime   as     dt
from   vispy      import app, gloo, visuals
from   PyQt5      import QtCore, QtGui, QtWidgets
from   OpenGL.GL  import *

import misc
import pipeline


_transform = vp.visuals.shaders.Function('''
    vec2 transform(vec2 p, vec2 a, vec2 b) {
        return a*p + b;
    }

    vec2 transform(vec2 p, float ax, float ay, float bx, float by) {
        return transform(p, vec2(ax, ay), vec2(bx, by));
    }
    ''')


class Generator(pipeline.Sampled):
    @property
    def paused(self):
        return not self._continueEvent.isSet()

    def __init__(self, fs, channels, **kwargs):
        # randomized channel parameters
        phases = np.random.uniform(size=(channels,1), low=0  , high=np.pi)
        freqs  = np.random.uniform(size=(channels,1), low=1  , high=5    )
        amps   = np.random.uniform(size=(channels,1), low=.05, high=.5   )
        amps2  = np.random.uniform(size=(channels,1), low=.05, high=.2   )

        # generate data as a (channels, ns) array.
        self._gen = lambda ns1, ns2: (
            amps * np.sin(2*np.pi*freqs*np.arange(ns1,ns2)/fs+phases)
            + amps2 * np.random.randn(channels, ns2-ns1))

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

    def start(self):
        self._continueEvent.set()
        if not self._thread.isAlive():
            self._thread.start()

    def pause(self):
        self._continueEvent.clear()

    def write(self, data, source=None):
        raise RuntimeError('Cannot write to a Generator')


class EpochPlot(pipeline.Node):
    _vertShaderSource = '''
        uniform vec3 u_bg_color;
        uniform vec2 u_plot_pos;
        uniform vec2 u_plot_size;
        uniform

        '''


class AnalogPlot(pipeline.Sampled):
    _vertShaderSource = '''
        uniform   vec2  u_plot_pos;       // unit in scene space
        uniform   vec2  u_plot_size;      // unit in scene space
        uniform   vec3  u_bg_color;
        uniform   vec3  u_colors[$channel_count];
        uniform   int   u_ns;
        uniform   int   u_ns_range;
        uniform   int   u_ns_fade;

        attribute float a_index;
        attribute float a_data;

        varying   float v_channel_index;
        varying   float v_sample_index;
        varying   vec2  v_pos_raw;
        varying   vec2  v_pos_channel;
        varying   vec2  v_pos_plot;
        varying   vec2  v_pos_scene;
        varying   vec2  v_pos_view;
        varying   vec2  v_pos_figure;
        varying   vec2  v_pos_canvas;
        varying   vec3  v_color;

        void main() {
            v_channel_index = floor(a_index / u_ns_range);
            v_sample_index  = mod(a_index, u_ns_range);

            v_pos_raw = vec2(v_sample_index, a_data);

            // all calculated coordinates below are in normalized space, i.e.:
            // -1 bottom-left to +1 top-right

            // channel
            v_pos_channel = $transform(v_pos_raw,
                2. / (u_ns_range - 1), 1,
                -1, 0);

            // plot (only current plot)
            v_pos_plot    = $transform(v_pos_channel,
                1, 1. / $channel_count,
                0, 1. - 2 * (v_channel_index + .5) / $channel_count);

            // scene (all plots, before pan & zoom)
            v_pos_scene   = $transform(v_pos_plot,
                u_plot_size,
                ((u_plot_pos + u_plot_size/2) * 2 - 1) * vec2(1,-1));

            // view (after pan & zoom)
            v_pos_view    = $transform_view(v_pos_scene);

            // figure (title and axes)
            v_pos_figure  = $transform_figure(v_pos_view);

            // canvas
            v_pos_canvas  = $transform_canvas(v_pos_figure);

            gl_Position   = vec4(v_pos_canvas, 0, 1);

            v_color       = u_colors[int(v_channel_index)];
        }
        '''

    _fragShaderSource = '''
        uniform vec3  u_bg_color;
        uniform int   u_ns;
        uniform int   u_ns_range;
        uniform int   u_ns_fade;

        varying float v_sample_index;
        varying float v_channel_index;
        varying vec2  v_pos_raw;
        varying vec2  v_pos_channel;
        varying vec2  v_pos_plot;
        varying vec2  v_pos_view;
        varying vec2  v_pos_figure;
        varying vec3  v_color;

        float fade_lin(float ns) {
            return (v_sample_index-ns)/u_ns_fade;
        }

        float fade_sin(float ns) {
            return sin(((v_sample_index-ns)/u_ns_fade*2-1)*3.1415/2)/2+.5;
        }

        float fade_pow3(float ns) {
            return pow((v_sample_index-ns)/u_ns_fade*2-1, 3)/2+.5;
        }

        void main() {
            float ns = mod(u_ns, u_ns_range);

            // discard the fragments between the channels
            // (emulate glMultiDrawArrays)
            if (0 < fract(v_channel_index))
                discard;

            // clip to view space
            if (1 < abs(v_pos_view.x) || 1 < abs(v_pos_view.y))
                discard;

            // discard unwritten samples in the first time window
            if (u_ns-1 <= v_sample_index)
                discard;

            // discard the fragment immediately after the last sample
            if (int(ns-1) == int(v_sample_index))
                discard;

            // calculate alpha
            float alpha = 1.;
            if (ns-1 <= v_sample_index && v_sample_index <= ns+u_ns_fade)
                alpha = fade_sin(ns);
            if (ns-1-u_ns_range <= v_sample_index &&
                    v_sample_index <= ns+u_ns_fade-u_ns_range)
                alpha = fade_sin(ns-u_ns_range);

            // set fragment color
            gl_FragColor = vec4(v_color*alpha + u_bg_color*(1-alpha), 1.);
            //gl_FragColor = vec4(v_color, alpha);
        }
        '''

    def __init__(self, figure, untPos=(0,0), untSize=(1,1), names=None,
            tsRange=10, tsFade=7, **kwargs):
        '''
        Args:
            figure:
            untPos: Position of the plot in unit scene space.
            untSize: Size of the plot in unit scene space.
            names:
            tsRange:
            tsFade:
        '''

        if (names is not None and not misc.iterable(names)
                and not callable(names)):
            raise TypeError('`names` should be None, iterable or callable')

        self._figure  = figure
        self._canvas  = figure._canvas
        self._untPos  = untPos
        self._untSize = untSize
        self._names   = names
        self._tsRange = tsRange
        self._tsFade  = tsFade
        self._lock    = threading.Lock()
        self._program = None
        self._texts   = []

        super().__init__(**kwargs)

        figure._plots += [self]

    def _configured(self, params):
        super()._configured(params)

        if misc.iterable(self._names) and len(self._names) != self._channels:
            raise ValueError('`names` should be the same size as number of'
                ' `channels`')

        # init local vars that depend on node configuration (fs and channels)
        self._colors     = np.random.uniform(size=(self._channels, 3),
            low=.3, high=1)
        self._zoomLevels = int(np.round(np.log2(self._fs/750)))+1
        self._zoomLevel  = self._zoomLevels-1
        self._fss        = [None]*self._zoomLevels
        self._indices    = [None]*self._zoomLevels
        self._buffers    = [None]*self._zoomLevels

        # init gpu program
        self._program    = vp.visuals.shaders.ModularProgram(
            self._vertShaderSource, self._fragShaderSource)

        self._program.vert['transform_view'  ] = self._figure._transformView
        self._program.vert['transform_figure'] = self._figure._transformFigure
        self._program.vert['transform_canvas'] = self._figure._transformCanvas
        self._program.vert['transform'       ] = _transform
        self._program.vert['channel_count'   ] = str(self._channels)

        self._program     ['u_bg_color'      ] = self._figure._bgColor
        self._program     ['u_plot_pos'      ] = self._untPos
        self._program     ['u_plot_size'     ] = self._untSize
        for i in range(self._channels):
            self._program ['u_colors[%d]' % i] = self._colors[i]

        # generate texts for each channel
        if self._names is None:
            self._texts = []
        else:
            self._texts = [None]*self._channels
            for i in range(self._channels):
                if callable(self._names):
                    name = self._names(i+1)
                else:
                    name = self._names[i]
                text = vp.visuals.TextVisual(name, bold=True, font_size=14,
                    color=self._figure._fgColor)
                text.anchors = ('right', 'center')
                self._texts[i] = text

        # make buffer and arrays for each zoom level
        for i in range(self._zoomLevels):
            ds       = 2**i
            fs       = self._fs / ds
            nsRange  = int(self._tsRange*fs)

            self._fss           [i] = fs
            self._indices       [i] = np.arange(self._channels*nsRange).reshape(
                (self._channels,-1))
            self._buffers       [i] = pipeline.CircularBuffer(size=nsRange)

            # internal pipeline
            self >> pipeline.DownsampleMinMax(ds=ds) >> self._buffers[i]

        self._updateProgram()
        self._updateTexts()

    def _updateProgram(self):
        if self._program is None: return

        level = self._zoomLevel
        prog  = self._program

        prog['u_ns'      ] = self._buffers[level].ns
        prog['u_ns_range'] = int(self._tsRange * self._fss[level])
        prog['u_ns_fade' ] = int(self._tsFade * self._fss[level])

        prog['a_index'   ] = self._indices[level].astype(np.float32)
        prog['a_data'    ] = self._buffers[level].data.astype(np.float32)

    def _updateTexts(self):
        if not self._texts: return

        fig = self._figure

        for i in range(self._channels):
            x = fig._pxlPos[0] + fig._pxlAxesSize[0] - 5
            y = ((fig._pxlPos[1] + fig._pxlViewSize[1]/self._channels*(i+.5)
                - fig._pxlViewCenter[1] + fig._pxlAxesSize[1]) * fig._zoom[1]
                - fig._pan[1]/2*fig._pxlViewSize[1] + fig._pxlViewCenter[1])
            self._texts[i].pos = (x, y)
            self._texts[i].visible = (fig._pxlAxesSize[1] < y - fig._pxlPos[1]
                < fig._pxlAxesSize[3] + fig._pxlViewSize[1])

    def canvasResize(self):
        if self._program is None: return

        for text in self._texts:
            text.transforms.configure(canvas=self._canvas,
                viewport=self._canvas.viewport)

        self._updateTexts()

    def figureZoom(self):
        if self._program is None: return

        self._updateTexts()

        level = self._zoomLevels - 1 - np.log2(self._figure._zoom[0])
        level = int(np.round(level))
        level = np.clip(level, 0, self._zoomLevels-1)

        if level != self._zoomLevel:
            with self._lock:
                self._zoomLevel = level
                self._updateProgram()

    def figurePan(self):
        if self._program is None: return

        self._updateTexts()

    def draw(self):
        if self._program is None: return

        with self._lock:
            self._program.draw('line_strip')

        for text in self._texts:
            text.draw()

    def write(self, data, source=None):
        super().write(data, source)
        with self._lock:
            buffer = self._buffers[self._zoomLevel]
            self._program['u_ns'  ] = buffer.ns
            self._program['a_data'] = buffer.data.astype(np.float32)


class Figure:
    @property
    def zoom(self):
        return self._zoom

    @zoom.setter
    def zoom(self, zoom):
        zoom = np.array(zoom)
        zoom = zoom.clip(1, 1000)
        self._zoom = zoom
        self._transformView['zoom'] = zoom
        for plot in self._plots:
            plot.figureZoom()
        self._canvas.update()

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
        for plot in self._plots:
            plot.figurePan()
        self._canvas.update()

    def __init__(self, canvas, untPos=(0,0), untSize=(1,1),
            pxlAxesSize=(30,15,15,15), bgColor=None, fgColor=None):
        '''
        Args:
            untPos: Position of figure in unit canvas space: (x, y).
            untSize: Size of figure in unit canvas space: (w, h).
            fgColor: Foreground color for border and texts. If None, defaults
                to canvas.fgColor.
            pxlAxesSize: Margins between the view and the figure in pixels:
                (l, t, r, b).
        '''

        if not isinstance(canvas, Canvas):
            raise TypeError('`canvas` should be instance of Canvas')

        self._canvas      = canvas
        self._untPos      = np.array(untPos)
        self._untSize     = np.array(untSize)
        self._pxlAxesSize = np.array(pxlAxesSize)
        self._bgColor     = canvas._bgColor if bgColor is None else bgColor
        self._fgColor     = canvas._fgColor if fgColor is None else fgColor

        self._plots       = []
        self._zoom        = np.array([1, 1])
        self._pan         = np.array([0, 0])

        self._transformView = vp.visuals.shaders.Function('''
            vec2 transform_view(vec2 pos) {
                return $transform(pos, $zoom, $pan);
            }''')
        self._transformView['transform'] = _transform
        self._transformView['zoom'] = self._zoom
        self._transformView['pan' ] = self._pan

        self._transformFigure = vp.visuals.shaders.Function('''
            vec2 transform_figure(vec2 pos) {
                return $transform(pos,
                    $view_size / $figure_size,
                    ($axes_size.xy - $axes_size.wz) / $figure_size
                    * vec2(1,-1));
            }''')
        self._transformFigure['transform'] = _transform
        self._transformFigure['axes_size'] = self._pxlAxesSize

        self._transformCanvas = vp.visuals.shaders.Function('''
            vec2 transform_canvas(vec2 pos) {
                return $transform(pos,
                    $figure_size / $canvas_size,
                    (($figure_pos + $figure_size/2) * 2 / $canvas_size - 1)
                    * vec2(1,-1));
            }''')
        self._transformCanvas['transform'] = _transform

        self._border = vp.visuals._BorderVisual(pos=(0,0), halfdim=(0,0),
            border_width=1, border_color=self._fgColor)
        # self._line = vp.visuals.LineVisual(np.array([[0, 0], [500,500]]))

        self._visuals = vp.visuals.CompoundVisual(
            [
            self._border,
            # self._line,
            ])

        self.canvasResize()

        canvas._figures += [self]

    def _pxl2nrmView(self, pxlDelta):
        # translate a delta (dx, dy) from pixels to normalized in view space
        # mainly used for panning
        return pxlDelta * 2 / self._pxlViewSize * [1, -1]

    def canvasResize(self):
        pxlCanvasSize = np.array(self._canvas.size)
        self._pxlPos  = self._untPos * pxlCanvasSize
        self._pxlSize = self._untSize * pxlCanvasSize
        self._pxlViewCenter = self._pxlPos + (self._pxlSize
            + self._pxlAxesSize[0:2] - self._pxlAxesSize[2:4])/2
        self._pxlViewSize   = (self._pxlSize - self._pxlAxesSize[0:2] -
            self._pxlAxesSize[2:4])

        self._visuals.transforms.configure(canvas=self._canvas,
            viewport=self._canvas.viewport)
        # self._line.transforms.configure(canvas=self, viewport=viewport)
        self._border.pos     = self._pxlViewCenter
        self._border.halfdim = self._pxlViewSize/2

        self._transformFigure['view_size'  ] = self._pxlViewSize
        self._transformFigure['figure_size'] = self._pxlSize

        self._transformCanvas['figure_pos' ] = self._pxlPos
        self._transformCanvas['figure_size'] = self._pxlSize
        self._transformCanvas['canvas_size'] = pxlCanvasSize

        for plot in self._plots:
            plot.canvasResize()

    def canvasMouseWheel(self, event):
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

    def canvasMousePress(self, event):
        if not self.isInside(event.pos): return

        if (event.button == 1 and 'Control' in event.modifiers):
            # start pan
            event.figure = self
            event.pan    = self.pan

    def canvasMouseMove(self, event):
        if (event.press_event and hasattr(event.press_event, 'figure')
                and event.press_event.figure == self
                and event.press_event.button == 1
                and 'Control' in event.press_event.modifiers):
            # pan
            delta = self._pxl2nrmView(event.pos - event.press_event.pos)
            self.pan = event.press_event.pan + delta

    def canvasMouseRelease(self, event):
        if (event.press_event and hasattr(event.press_event, 'figure')
                and event.press_event.figure == self
                and event.press_event.button == 1
                and 'Control' in event.press_event.modifiers):
            # stop pan
            delta = self._pxl2nrmView(event.pos-event.press_event.pos)
            self.pan = event.press_event.pan + delta

    def canvasKeyRelease(self, event):
        if 'Control' in event.modifiers and event.key == '0':
            self.zoom = [1, 1]
            self.pan  = [0, 0]

    def isInside(self, pxl):
        return ((self._pxlPos < pxl) & (pxl < self._pxlPos+self._pxlSize)).all()

    def draw(self):
        for plot in self._plots:
            plot.draw()
        self._visuals.draw()


class Canvas(vp.app.Canvas):
    @property
    def viewport(self):
        return (0, 0, *self.physical_size)

    def __init__(self, bgColor=(0,0,.2), fgColor=(0,1,1)):

        super().__init__(vsync=True, config=dict(samples=4))

        self._bgColor   = bgColor
        self._fgColor   = fgColor
        self._figures   = []

        vp.gloo.set_state(clear_color=self._bgColor, blend=True,
            depth_test=False)

        self._fpsText = vp.visuals.TextVisual('0.0', bold=True, color=fgColor,
            pos=(1,1), font_size=8)
        self._fpsText.anchors = ('left', 'top')
        self._mouseText = vp.visuals.TextVisual('', bold=True, color=fgColor,
            pos=(30,1), font_size=8)
        self._mouseText.anchors = ('left', 'top')

        self._visuals = vp.visuals.CompoundVisual(
            [self._fpsText,
            # self._mouseText,
            ])

        self._updateLast  = None
        self._updateTimes = []    # keep last update times for measuring FPS
        self._startTime   = dt.datetime.now()
        self._timer       = vp.app.Timer('auto', connect=self.on_timer)

        self._timer.start()

    def on_resize(self, event):
        self.context.set_viewport(*self.viewport)
        self._visuals.transforms.configure(canvas=self, viewport=self.viewport)

        for figure in self._figures:
            figure.canvasResize()

    def on_mouse_wheel(self, event):
        for figure in self._figures:
            figure.canvasMouseWheel(event)

    def on_mouse_press(self, event):
        for figure in self._figures:
            figure.canvasMousePress(event)

    def on_mouse_move(self, event):
        self._mouseText.text = str(event.pos)
        for figure in self._figures:
            figure.canvasMouseMove(event)

    def on_mouse_release(self, event):
        for figure in self._figures:
            figure.canvasMouseRelease(event)

    def on_key_release(self, event):
        for figure in self._figures:
            figure.canvasKeyRelease(event)

        if event.key == 'space' and hasattr(self, 'onSpace'):
            self.onSpace()

        # elif 'Control' in event.modifiers and event.key == 'left':
        #     if self.pan_view(self._viewSize/10*[1,0]):
        #         self.update()
        # elif 'Control' in event.modifiers and event.key == 'right':
        #     if self.pan_view(self._viewSize/10*[-1,0]):
        #         self.update()
        # elif 'Control' in event.modifiers and event.key == 'up':
        #     if self.pan_view(self._viewSize/10*[0,1]):
        #         self.update()
        # elif 'Control' in event.modifiers and event.key == 'down':
        #     if self.pan_view(self._viewSize/10*[0,-1]):
        #         self.update()

    def on_timer(self, event):
        self.update()

    def on_draw(self, event):
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

        vp.gloo.clear()
        # glEnable(GL_LINE_STIPPLE)
        # glLineStipple(50, 0b1010101010101010)
        # self._line.draw()
        # glDisable(GL_LINE_STIPPLE)
        # glLineWidth(1)
        for figure in self._figures:
            figure.draw()
        self._visuals.draw()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.canvas = Canvas()
        self.figure = Figure(self.canvas, untPos=(0,0), untSize=(1,1))
        self.plot   = AnalogPlot(self.figure, untSize=(1,1), names=str)
        self.plot2   = AnalogPlot(self.figure, untSize=(1,1), names=str)
        # self.plot2  = AnalogPlot(self.figure, untPos=(.5,.5),
        #     untSize=(.5,.5), names=str)

        self.generator = Generator(fs=31.25e3, channels=15)
        self.generator >> self.plot

        self.generator.start()

        self.canvas.onSpace = self.onSpace

        self.button = QtWidgets.QPushButton('Test')

        mainLayout = QtWidgets.QHBoxLayout()
        mainLayout.setContentsMargins(0, 0, 0, 0)
        mainLayout.addWidget(self.canvas.native)
        # mainLayout.addWidget(self.button)

        self.setLayout(mainLayout)

    def onSpace(self):
        if self.generator.paused:
            self.generator.start()
        else:
            self.generator.pause()


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
