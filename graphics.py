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


class Canvas(vp.app.Canvas):
    @property
    def bgColor(self):
        return self._bgColor

    @property
    def fgColor(self):
        return self._fgColor

    @property
    def viewport(self):
        return (0, 0, *self.physical_size)

    def __init__(self, bgColor=(1,1,1), fgColor=(.2,.2,.2)):

        super().__init__(vsync=True, config=dict(samples=4))

        self._bgColor   = bgColor
        self._fgColor   = fgColor
        self._figures   = []

        vp.gloo.set_state(clear_color=bgColor, blend=True, depth_test=False)

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
            figure.canvasResized()

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

        if event.key == 'space' and hasattr(self, 'spacePressed'):
            self.spacePressed()

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


class Figure:
    @property
    def bgColor(self):
        if self._bgColor is None:
            return self._canvas.bgColor
        else:
            return self._bgColor

    @property
    def fgColor(self):
        if self._fgColor is None:
            return self._canvas.fgColor
        else:
            return self._fgColor

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
            plot.figureZoomed()
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
            plot.figurePanned()
        self._canvas.update()

    def __init__(self, canvas=None, untPos=(0,0), untSize=(1,1),
            pxlAxesSize=(60,15,15,15), bgColor=None, fgColor=None,
            borderWidth=1):
        '''
        Args:
            untPos: Position of figure in unit canvas space: (x, y).
            untSize: Size of figure in unit canvas space: (w, h).
            fgColor: Foreground color for border and texts. If None, defaults
                to canvas.fgColor.
            pxlAxesSize: Margins between the view and the figure in pixels:
                (l, t, r, b).
        '''

        # allow childs to mix with other classes
        super().__init__()

        if not isinstance(canvas, Canvas):
            raise TypeError('`canvas` should be instance of Canvas')

        self._canvas      = canvas
        self._untPos      = np.array(untPos)
        self._untSize     = np.array(untSize)
        self._pxlAxesSize = np.array(pxlAxesSize)
        self._bgColor     = bgColor
        self._fgColor     = fgColor

        self._plots       = []
        self._zoom        = np.array([1, 1])
        self._pan         = np.array([0, 0])

        self._transformView = vp.visuals.shaders.Function('''
            vec2 transform_view(vec2 pos) {
                return $transform(pos, $zoom, $pan);
            }''')
        self._transformView['transform'] = _glslTransform
        self._transformView['zoom'] = self._zoom
        self._transformView['pan' ] = self._pan

        self._transformFigure = vp.visuals.shaders.Function('''
            vec2 transform_figure(vec2 pos) {
                return $transform(pos,
                    $view_size / $figure_size,
                    ($axes_size.xy - $axes_size.wz) / $figure_size
                    * vec2(1,-1));
            }''')
        self._transformFigure['transform'] = _glslTransform
        self._transformFigure['axes_size'] = self._pxlAxesSize

        self._transformCanvas = vp.visuals.shaders.Function('''
            vec2 transform_canvas(vec2 pos) {
                return $transform(pos,
                    $figure_size / $canvas_size,
                    (($figure_pos + $figure_size/2) * 2 / $canvas_size - 1)
                    * vec2(1,-1));
            }''')
        self._transformCanvas['transform'] = _glslTransform

        self._border = vp.visuals._BorderVisual(pos=(0,0), halfdim=(0,0),
            border_width=borderWidth, border_color=self.fgColor)
        # self._line = vp.visuals.LineVisual(np.array([[0, 0], [500,500]]))

        self._visuals = vp.visuals.CompoundVisual(
            [
            self._border,
            # self._line,
            ])

        self.canvasResized()

        canvas._figures += [self]

    def _pxl2nrmView(self, pxlDelta):
        # translate a delta (dx, dy) from pixels to normalized in view space
        # mainly used for panning
        return pxlDelta * 2 / self._pxlViewSize * [1, -1]

    def canvasResized(self):
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
            plot.canvasResized()

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


class Scope(Figure, pipeline.Sampled):
    @property
    def ns(self):
        return self._ns

    @property
    def ts(self):
        if self._fs is None:
            # node not configured yet
            return 0
        elif self._ns == 0:
            return 0
        else:
            return (self._ns - 1) / self._fs

    @property
    def tsRange(self):
        return self._tsRange

    @tsRange.setter
    def tsRange(self, tsRange):
        raise NotImplementedError()
        # self._tsRange = tsRange

    @property
    def tsFade(self):
        return self._tsFade

    def __init__(self, tsRange=10, tsFade=5, **kwargs):
        super().__init__(**kwargs)

        self._ns      = 0
        self._tsRange = tsRange
        self._tsFade  = tsFade

    def _configured(self, params):
        super()._configured(params)

        # inform plots of change in fs
        for plot in self._plots:
            if hasattr(plot, 'fsChanged'):
                plot.fsChanged()

    def write(self, data, source=None):
        super().write(data, source)

        # increment total number of samples
        self._ns += data.shape[1]

        # inform plots of change in ns/ts
        for plot in self._plots:
            if hasattr(plot, 'tsChanged'):
                plot.tsChanged()


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
        }
        '''

    _fragShaderSource = '''
        uniform   vec2  u_plot_pos;       // unit in scene space
        uniform   vec2  u_plot_size;      // unit in scene space
        uniform   vec3  u_bg_color;
        uniform   vec3  u_colors[$channel_count];
        uniform   int   u_ns;
        uniform   int   u_ns_range;
        uniform   int   u_ns_fade;

        varying   float v_sample_index;
        varying   float v_channel_index;
        varying   vec2  v_pos_raw;
        varying   vec2  v_pos_channel;
        varying   vec2  v_pos_plot;
        varying   vec2  v_pos_scene;
        varying   vec2  v_pos_view;
        varying   vec2  v_pos_figure;
        varying   vec2  v_pos_canvas;

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

            // calculate fading factor
            float fade = 1.;
            if (ns-1 <= v_sample_index && v_sample_index <= ns+u_ns_fade)
                fade = fade_sin(ns);
            if (ns-1-u_ns_range <= v_sample_index &&
                    v_sample_index <= ns+u_ns_fade-u_ns_range)
                fade = fade_sin(ns-u_ns_range);

            // set fragment color
            vec3 color = u_colors[int(v_channel_index)];
            gl_FragColor = vec4(color*fade + u_bg_color*(1-fade), 1.);
            //gl_FragColor = vec4(color, fade);
        }
        '''

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

    def __init__(self, figure, untPos=(0,0), untSize=(1,1), names=None,
            colors=None, **kwargs):
        '''
        Args:
            figure:
            untPos: Position of the plot in unit scene space.
            untSize: Size of the plot in unit scene space.
            names:
            tsRange:
            tsFade:
        '''

        if not isinstance(figure, Scope):
            raise TypeError('`figure` should be an instance of Scope')

        if (names is not None and not misc.iterable(names)
                and not callable(names)):
            raise TypeError('`names` should be None, iterable or callable')

        self._figure  = figure
        self._canvas  = figure._canvas
        self._untPos  = untPos
        self._untSize = untSize
        self._names   = names
        self._colors  = colors

        self._lock    = threading.Lock()
        self._program = None
        self._texts   = []

        figure._plots += [self]

        super().__init__(**kwargs)

    def _configured(self, params):
        super()._configured(params)

        if misc.iterable(self._names) and len(self._names) != self._channels:
            raise ValueError('`names` should be the same size as number of'
                ' `channels`')

        if self._colors is not None and len(self._colors) != self._channels:
            raise ValueError('`colors` should be the same size as number of'
                ' `channels`')

        # init local vars that depend on node configuration (fs and channels)
        if self._colors == None:
            self._colors = self._defaultColors[
                np.arange(self._channels) % len(self._defaultColors)]
            # self._colors  = np.random.uniform(size=(self._channels, 3),
            #     low=.3, high=1)
        self._zooms   = int(np.round(np.log2(
                        self._fs/7500*self._figure.tsRange)))+1
        self._zoom    = self._zooms-1
        self._fss     = [None]*self._zooms
        self._indices = [None]*self._zooms
        self._buffers = [None]*self._zooms

        # init gpu program
        self._program    = vp.visuals.shaders.ModularProgram(
            self._vertShaderSource, self._fragShaderSource)

        self._program.vert['transform_view'  ] = self._figure._transformView
        self._program.vert['transform_figure'] = self._figure._transformFigure
        self._program.vert['transform_canvas'] = self._figure._transformCanvas
        self._program.vert['transform'       ] = _glslTransform
        self._program.vert['channel_count'   ] = str(self._channels)
        self._program.frag['channel_count'   ] = str(self._channels)

        self._program     ['u_bg_color'      ] = self._figure.bgColor
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
                    color=self._colors[i]) #self._figure.fgColor)
                text.anchors = ('right', 'center')
                self._texts[i] = text

        # make buffer and arrays for each zoom level
        for i in range(self._zooms):
            ds       = 2**i
            fs       = self._fs / ds
            nsRange  = int(self._figure.tsRange*fs)

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

        zoom = self._zoom
        prog = self._program

        prog['u_ns'      ] = self._buffers[zoom].ns
        prog['u_ns_range'] = int(self._figure.tsRange * self._fss[zoom])
        prog['u_ns_fade' ] = int(self._figure.tsFade  * self._fss[zoom])

        prog['a_index'   ] = self._indices[zoom].astype(np.float32)
        prog['a_data'    ] = self._buffers[zoom].data.astype(np.float32)

    def _updateTexts(self):
        if not self._texts: return

        fig = self._figure

        for i in range(self._channels):
            x = fig._pxlPos[0] + fig._pxlAxesSize[0] - 5
            y = (fig._pxlViewSize[1] * ((i+.5)/self._channels*self._untSize[1]
                - .5 + self._untPos[1]) * fig._zoom[1]
                - fig._pan[1]/2 * fig._pxlViewSize[1] + fig._pxlViewCenter[1])
            self._texts[i].pos = (x, y)
            self._texts[i].visible = (fig._pxlAxesSize[1] < y - fig._pxlPos[1]
                < fig._pxlAxesSize[3] + fig._pxlViewSize[1])

    def canvasResized(self):
        if self._program is None: return

        for text in self._texts:
            text.transforms.configure(canvas=self._canvas,
                viewport=self._canvas.viewport)

        self._updateTexts()

    def figureZoomed(self):
        if self._program is None: return

        self._updateTexts()

        zoom = self._zooms - 1 - np.log2(self._figure._zoom[0])
        zoom = int(np.round(zoom))
        zoom = np.clip(zoom, 0, self._zooms-1)

        if zoom != self._zoom:
            with self._lock:
                self._zoom = zoom
                self._updateProgram()

    def figurePanned(self):
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
            buffer = self._buffers[self._zoom]
            self._program['u_ns'  ] = buffer.ns
            self._program['a_data'] = buffer.data.astype(np.float32)


class AnalogGenerator(pipeline.Sampled):
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
        uniform   vec2  u_plot_pos;
        uniform   vec2  u_plot_size;
        //uniform   vec3  u_bg_color;
        uniform   vec4  u_color;
        uniform   float u_fs;
        uniform   float u_ts;
        uniform   float u_ts_range;
        uniform   float u_ts_fade;

        attribute float a_index;
        attribute float a_data;
        attribute float a_data2;

        varying   float v_epoch_index;
        varying   float v_epoch_ts;
        varying   vec2  v_pos_raw;
        varying   vec2  v_pos_plot;
        varying   vec2  v_pos_scene;
        varying   vec2  v_pos_view;
        varying   vec2  v_pos_figure;
        varying   vec2  v_pos_canvas;

        void main() {
            v_epoch_index   = floor(a_index/4);
            v_epoch_ts      = a_data;

            float ts_window = floor(u_ts / u_ts_range) * u_ts_range;
            bool  start     = mod(int(a_index), 2) == 0;
            bool  stop      = mod(int(a_index), 2) == 1;
            bool  ghost     = mod(int(v_epoch_index), 2) == 1;

            // unstopped epochs that their start timestamp is larger than
            // the current timestamp should be discarded in fragment shader
            if (start && a_data2<0 && u_ts<v_epoch_ts)
                v_epoch_ts  = -1;

            // set stop timestamp of unstopped epochs to the current timestamp
            if (stop && v_epoch_ts<0 && 0<=a_data2 && a_data2<=u_ts)
                v_epoch_ts  = u_ts;

            v_pos_raw       = vec2(v_epoch_ts - ts_window,
                mod(floor(a_index/2), 2));

            // ghosting for screen wrapping
            if (ghost && v_epoch_ts>=0)
                v_pos_raw.x += u_ts_range;

            // all calculated coordinates below are in normalized space, i.e.:
            // -1 bottom-left to +1 top-right

            // plot (only current plot)
            v_pos_plot       = $transform(v_pos_raw,
                2 / (u_ts_range - 1/u_fs), 2,
                -1, -1);

            // scene (all plots, before pan & zoom)
            v_pos_scene      = $transform(v_pos_plot,
                u_plot_size,
                ((u_plot_pos + u_plot_size/2) * 2 - 1) * vec2(1,-1));

            // view (after pan & zoom)
            v_pos_view       = $transform_view(v_pos_scene);

            // figure (title and axes)
            v_pos_figure     = $transform_figure(v_pos_view);

            // canvas
            v_pos_canvas     = $transform_canvas(v_pos_figure);

            gl_Position      = vec4(v_pos_canvas, 0, 1);
        }
        '''

    _fragShaderSource = '''
        uniform   vec2  u_plot_pos;
        uniform   vec2  u_plot_size;
        //uniform   vec3  u_bg_color;
        uniform   vec4  u_color;
        uniform   float u_ts;
        uniform   float u_ts_range;
        uniform   float u_ts_fade;

        varying   float v_epoch_index;
        varying   float v_epoch_ts;
        varying   vec2  v_pos_raw;
        varying   vec2  v_pos_plot;
        varying   vec2  v_pos_scene;
        varying   vec2  v_pos_view;
        varying   vec2  v_pos_figure;
        varying   vec2  v_pos_canvas;

        //float fade_lin(float ns) {
        //    return (v_sample_index-ns)/u_ns_fade;
        //}

        float fade_sin(float ts) {
            return sin(((v_pos_raw.x-ts)/u_ts_fade*2-1)*3.1415/2)/2+.5;
        }

        //float fade_pow3(float ns) {
        //    return pow((v_sample_index-ns)/u_ns_fade*2-1, 3)/2+.5;
        //}

        void main() {
            float ts_window = floor(u_ts / u_ts_range) * u_ts_range;
            float ts = u_ts - ts_window;

            // discard the fragments between epochs
            if (0 < fract(v_epoch_index))
                discard;

            // clip to view space
            if (1 < abs(v_pos_view.x) || 1 < abs(v_pos_view.y))
                discard;

            // discard unwritten samples in the first time window
            if (v_epoch_ts < u_ts-u_ts_range || u_ts < v_epoch_ts)
                discard;

            // calculate fading factor
            float fade = 1.;
            if (ts <= v_pos_raw.x && v_pos_raw.x <= ts+u_ts_fade)
                fade = fade_sin(ts);
            if (ts-u_ts_range <= v_pos_raw.x &&
                    v_pos_raw.x <= ts+u_ts_fade-u_ts_range)
                fade = fade_sin(ts-u_ts_range);

            // set fragment color
            //gl_FragColor = vec4(u_color.rgb*fade + u_bg_color*(1-fade), 1.);
            gl_FragColor = u_color;
            gl_FragColor.a *= fade;
            //gl_FragColor.r += ($rand(v_pos_raw+1)-.5)/255;
            //gl_FragColor.g += ($rand(v_pos_raw+2)-.5)/255;
            //gl_FragColor.b += ($rand(v_pos_raw+3)-.5)/255;
            gl_FragColor.a += ($rand(v_pos_raw+4)-.5)/255;
        }
        '''

    def __init__(self, figure, untPos=(0,0), untSize=(1,1), name=None,
            color=(0,1,0,.6), **kwargs):

        if not isinstance(figure, Scope):
            raise TypeError('`figure` should be an instance of Scope')

        if name is not None and not isinstance(name, str):
            raise TypeError('`names should be None or a string`')

        self._figure  = figure
        self._canvas  = figure._canvas
        self._untPos  = untPos
        self._untSize = untSize
        self._color   = color
        self._name    = name
        self._lock    = threading.Lock()
        self._cache   = 100*8    # cache 100 epochs
        self._pointer = 0
        self._partial = None
        self._index   = np.arange(self._cache, dtype=np.float32)
        self._data    = np.zeros (self._cache, dtype=np.float32)-1

        self._program = vp.visuals.shaders.ModularProgram(
            self._vertShaderSource, self._fragShaderSource)

        self._program.vert['transform_view'  ] = self._figure._transformView
        self._program.vert['transform_figure'] = self._figure._transformFigure
        self._program.vert['transform_canvas'] = self._figure._transformCanvas
        self._program.vert['transform'       ] = _glslTransform
        self._program.frag['rand'            ] = _glslRand

        self._program     ['u_plot_pos'      ] = self._untPos
        self._program     ['u_plot_size'     ] = self._untSize
        # self._program     ['u_bg_color'      ] = self._figure.bgColor
        self._program     ['u_color'         ] = self._color
        self._program     ['u_ts'            ] = 0
        self._program     ['u_ts_range'      ] = self._figure.tsRange
        self._program     ['u_ts_fade'       ] = self._figure.tsFade

        self._program     ['a_index'         ] = self._index
        self._program     ['a_data'          ] = self._data
        self._program     ['a_data2'         ] = self._data.reshape(
                                                (-1,2))[:,::-1].ravel()

        if name is None:
            self._text = None
        else:
            self._text = vp.visuals.TextVisual(name, bold=True, font_size=10,
                rotation=0, color=self._color) #self._figure.fgColor)
            self._text.anchors = ('right', 'center')

        self._updateText()

        figure._plots += [self]

        super().__init__(**kwargs)

    def _updateText(self):
        if not self._text: return

        fig = self._figure

        x = fig._pxlPos[0] + fig._pxlAxesSize[0] - 5
        y = (fig._pxlViewSize[1] * (self._untSize[1]/2
            - .5 + self._untPos[1]) * fig._zoom[1]
            - fig._pan[1]/2 * fig._pxlViewSize[1] + fig._pxlViewCenter[1])
        self._text.pos = (x, y)
        self._text.visible = (fig._pxlAxesSize[1] < y - fig._pxlPos[1]
            < fig._pxlAxesSize[3] + fig._pxlViewSize[1])

    def canvasResized(self):
        if self._text:
            self._text.transforms.configure(canvas=self._canvas,
                viewport=self._canvas.viewport)

        self._updateText()

    def figureZoomed(self):
        self._updateText()

    def figurePanned(self):
        self._updateText()

    def fsChanged(self):
        self._program['u_fs'] = self._figure._fs

    def tsChanged(self):
        self._program['u_ts'] = self._figure.ts

    def draw(self):
        with self._lock:
            self._program.draw('triangle_strip')

        self._text.draw()

    def write(self, data, source=None):
        super().write(data, source)

        if not isinstance(data, np.ndarray): data = np.array(data)
        if data.ndim == 0: data = data[np.newaxis]
        if data.ndim != 1: raise ValueError('`data` should be 1D')

        # when last added epoch has been partial, complete the epoch by adding
        # its start timestamp to the beginning of current data
        if self._partial is not None:
            data = np.insert(data, 0, self._partial)
            self._partial = None

        # when last epoch in the current data is partial, save its start
        # timestamp for the next write operation (see above)
        if len(data)%2 != 0:
            self._partial = data[-1]
            data = np.append(data, -1)

        # repeat each timestamp 4 times, 2 for top and bottom vertex of
        # the epoch rectangle and 2 for ghosting and screen wrapping
        data    = data.reshape((-1,2)).repeat(4,axis=0).ravel()
        # write data to (circular) buffer
        window  = np.arange(self._pointer, self._pointer+len(data))
        window %= self._cache
        self._data[window] = data

        if self._partial is None:
            self._pointer += len(data)
        else:
            self._pointer += len(data) - 8

        with self._lock:
            self._program['a_data' ] = self._data
            # swap epoch starts and stops
            self._program['a_data2'] = self._data.reshape(
                (-1,2))[:,::-1].ravel()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.canvas     = Canvas()
        self.scope      = Scope(canvas=self.canvas, untPos=(0,0),
            untSize=(1,1))
        self.analogPlot = AnalogPlot(self.scope, untPos=(0,0), untSize=(1,.9),
            names=str)
        self.targetPlot = EpochPlot(self.scope, untPos=(0,.9), untSize=(1,.05),
            name='Target', color=(0,1,0,.7))
        self.pumpPlot   = EpochPlot(self.scope, untPos=(0,.95), untSize=(1,.05),
            name='Pump', color=(0,0,1,.7))


        self.targetPlot.write([2,4])
        self.analogGenerator = AnalogGenerator(fs=31.25e3, channels=15)
        self.analogGenerator >> self.scope >> self.analogPlot
        self.analogGenerator.start()

        self.canvas.spacePressed = self.spacePressed

        self.button = QtWidgets.QPushButton('Test')

        mainLayout = QtWidgets.QHBoxLayout()
        mainLayout.setContentsMargins(0, 0, 0, 0)
        mainLayout.addWidget(self.canvas.native)
        # mainLayout.addWidget(self.button)

        self.setLayout(mainLayout)

    def spacePressed(self):
        self.targetPlot.write(self.scope.ts)
        self.pumpPlot.write(self.scope.ts+1)
        # if self.analogGenerator.paused:
        #     self.analogGenerator.start()
        # else:
        #     self.analogGenerator.pause()


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
