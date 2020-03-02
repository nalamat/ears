'''GPU accelerated graphics and plotting using vispy.


This file is part of the EARS project: https://github.com/nalamat/ears
Copyright (C) 2017-2020 Nima Alamatsaz <nima.alamatsaz@gmail.com>
Copyright (C) 2017-2020 NESH Lab <ears.software@gmail.com>
Distributed under GNU GPLv3. See LICENSE.txt for more info.
'''

import math
import time
import functools
import threading
import numpy      as     np
import scipy      as     sp
import vispy      as     vp
import datetime   as     dt
from   scipy      import signal
from   vispy      import app, gloo, visuals
from   PyQt5      import QtCore, QtGui, QtWidgets
from   ctypes     import sizeof, c_void_p, c_uint, c_float
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
            pxlMargins=(0,0,0,0), bgColor=None, fgColor=None, **kwargs):
        '''
        Args:
            untPos (2 floats): Position of container in unit parent space:
                (x, y).
            untSize (2 floats): Size of container in unit parent space: (w, h).
            pxlMargins (4 floats): Margins between the view and the figure in
                pixels: (l, t, r, b).
            bgColor (3 floats): Backgoround color for the container. If None,
                defaults to parent.bgColor.
            fgColor (3 floats): Foreground color for border and texts. If None,
                defaults to parent.fgColor.
        '''

        # allow multiple inheritance and mixing with other classes
        super().__init__(**kwargs)

        self._parent     = parent
        self._untPos     = np.array(untPos)
        self._untSize    = np.array(untSize)
        self._pxlMargins = np.array(pxlMargins)
        self._pxlPos     = np.array([0, 0])
        self._pxlSize    = np.array([1, 1])
        self._bgColor    = bgColor
        self._fgColor    = fgColor
        self._items      = []

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
            - parent._pxlMargins[0:2] - parent._pxlMargins[2:4])
            + parent._pxlMargins[0:2])
        self._pxlSize = self._untSize * (parent._pxlSize
            - parent._pxlMargins[0:2] - parent._pxlMargins[2:4])

        self._callItems('on_resize', *args, **kwargs)

    _funcs = ['on_mouse_wheel', 'on_mouse_press', 'on_mouse_move',
        'on_mouse_release', 'on_key_release', 'draw']
    for func in _funcs:
        exec('def %(func)s(self, *args, **kwargs):\n'
            '    self._callItems("%(func)s", *args, **kwargs)' % {'func':func})


class Canvas(Container, vp.app.Canvas):
    @property
    def canvas(self):
        return self

    @property
    def viewport(self):
        return (0, 0, *self.physical_size)

    def __init__(self, bgColor=(1,1,1), fgColor=(.3,.3,.3), **kwargs):

        super().__init__(bgColor=bgColor, fgColor=fgColor,
            vsync=True, config=dict(samples=1), **kwargs)

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
        self._pxlPos  = np.array([0,0])
        self._pxlSize = np.array(self.size)
        self._callItems('on_resize', event)

    def on_mouse_move(self, event):
        self._mouseText.text = str(event.pos)
        super().on_mouse_move(event)

    def on_key_release(self, event):
        if hasattr(self, 'keyReleased'):
            self.keyReleased(event)
        super().on_key_release(event)

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
        # glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        # glEnable(GL_LINE_STIPPLE)
        # glLineStipple(50, 0b1010101010101010)
        # self._line.draw()
        # glDisable(GL_LINE_STIPPLE)
        # glLineWidth(1)
        self.draw()
        self._visuals.draw()
        # self._border._program.draw(self._border._vshare.draw_mode,
        #                    self._border._vshare.index_buffer)
        # drawTest()


class Figure(Container):
    @property
    def zoom(self):
        return self._zoom

    @zoom.setter
    def zoom(self, zoom):
        zoom = np.array(zoom)
        zoom = zoom.clip(1, 1000)
        self._zoom = zoom
        self._transformView['zoom'] = zoom
        self._callItems('zoomed')
        self.canvas.update()

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
        self._callItems('panned')
        self.canvas.update()

    def __init__(self, borderWidth=1, **kwargs):
        '''
        Args:
            borderWidth (float): Width of the border drawn around figure's view
                box in pixels. Defaults to 1.
        '''

        # allow childs to mix with other classes
        super().__init__(**kwargs)

        if not isinstance(self._parent, Container):
            raise TypeError('`parent` should be a Container')

        self._pxlViewCenter = np.array([0, 0])
        self._pxlViewSize   = np.array([1, 1])
        self._zoom          = np.array([1, 1])
        self._pan           = np.array([0, 0])

        # setup glsl transformations for child plot items
        self._transformView = vp.visuals.shaders.Function('''
            vec2 transform_view(vec2 pos) {
                return $transform(pos, $zoom, $pan);
            }''')
        self._transformView['transform'] = _glslTransform
        self._transformView['zoom'     ] = self._zoom
        self._transformView['pan'      ] = self._pan

        self._transformFigure = vp.visuals.shaders.Function('''
            vec2 transform_figure(vec2 pos) {
                return $transform(pos,
                    $view_size / $figure_size,
                    ($margins.xy - $margins.wz) / $figure_size
                    * vec2(1,-1));
            }''')
        self._transformFigure['transform'] = _glslTransform
        self._transformFigure['margins'  ] = self._pxlMargins

        self._transformCanvas = vp.visuals.shaders.Function('''
            vec2 transform_canvas(vec2 pos) {
                return $transform(pos,
                    $figure_size / $canvas_size,
                    (($figure_pos + $figure_size/2) * 2 / $canvas_size - 1)
                    * vec2(1,-1));
            }''')
        self._transformCanvas['transform'] = _glslTransform

        # a border around the figure
        self._border = vp.visuals._BorderVisual(pos=(0,0), halfdim=(0,0),
            border_width=borderWidth, border_color=self.fgColor)
        # self._line = vp.visuals.LineVisual(np.array([[0, 0], [500,500]]))

        # bundle all visuals for easier management
        self._visuals = vp.visuals.CompoundVisual(
            [
            self._border,
            # self._line,
            ])

    def _pxl2nrmView(self, pxlDelta):
        # translate a delta (dx, dy) from pixels to normalized in view space
        # mainly used for panning
        return pxlDelta * 2 / self._pxlViewSize * [1, -1]

    def addItem(self, item):
        if not isinstance(item, Plot):
            raise TypeError('`item` should be a Plot')

        super().addItem(item)

    def on_resize(self, event):
        parent = self._parent

        self._pxlPos = (parent._pxlPos + self._untPos * (parent._pxlSize
            - parent._pxlMargins[0:2] - parent._pxlMargins[2:4])
            + parent._pxlMargins[0:2])
        self._pxlSize = self._untSize * (parent._pxlSize
            - parent._pxlMargins[0:2] - parent._pxlMargins[2:4])

        self._pxlViewCenter = self._pxlPos + (self._pxlSize
            + self._pxlMargins[0:2] - self._pxlMargins[2:4])/2
        self._pxlViewSize   = (self._pxlSize - self._pxlMargins[0:2] -
            self._pxlMargins[2:4])

        self._visuals.transforms.configure(canvas=self.canvas,
            viewport=self.canvas.viewport)
        # self._border.transforms.configure(canvas=self.canvas,
        #     viewport=self.canvas.viewport)
        self._border.pos     = self._pxlViewCenter
        self._border.halfdim = self._pxlViewSize/2

        self._transformFigure['view_size'  ] = self._pxlViewSize
        self._transformFigure['figure_size'] = self._pxlSize

        self._transformCanvas['figure_pos' ] = self._pxlPos
        self._transformCanvas['figure_size'] = self._pxlSize
        self._transformCanvas['canvas_size'] = self.canvas.size

        self._callItems('on_resize', event)

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

    def draw(self):
        super().draw()
        self._visuals.draw()


class Scope(Figure, pipeline.Sampled):
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
        '''
        Args:
            tsRange (float): Range of the active time window.
            tsFade (float): Range of fading the previous time window data.
        '''

        self._tsRange = tsRange
        self._tsFade  = tsFade

        super().__init__(**kwargs)

    def _written(self, data, source):
        super()._written(data, source)


class Plot:
    def __init__(self, figure=None, untPos=(0,0), untSize=(1,1), **kwargs):
        '''
        Args:
            figure:
            untPos: Position of the plot in unit scene space.
            untSize: Size of the plot in unit scene space.
        '''

        if not isinstance(figure, Figure):
            raise TypeError('`figure` should be an instance of Figure')

        self._figure  = figure
        self._canvas  = figure.canvas
        self._untPos  = untPos
        self._untSize = untSize

        self._lock    = threading.Lock()
        self._program = vp.visuals.shaders.ModularProgram(
            self._vertShaderSource, self._fragShaderSource)

        # init gpu program
        self._program.vert['transform_view'  ] = figure._transformView
        self._program.vert['transform_figure'] = figure._transformFigure
        self._program.vert['transform_canvas'] = figure._transformCanvas
        self._program.vert['transform'       ] = _glslTransform

        self._program     ['u_bg_color'      ] = self._figure.bgColor
        self._program     ['u_plot_pos'      ] = self._untPos
        self._program     ['u_plot_size'     ] = self._untSize

        figure.addItem(self)

        super().__init__(**kwargs)

    def draw(self, mode, smooth=False):
        if self._program is None: return

        if smooth: glEnable(GL_LINE_SMOOTH)
        with self._lock:
            self._program.draw(mode)
        if smooth: glDisable(GL_LINE_SMOOTH)


class AnalogPlot(Plot, pipeline.Sampled):
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
            //if (u_ns-1 <= v_sample_index)
                //discard;

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

    def __init__(self, names=None, colors=None, **kwargs):
        '''
        Args:
            names:
            colors:
        '''

        if not isinstance(kwargs.get('figure',None), Scope):
            raise TypeError('`figure` should be an instance of Scope')

        if (names is not None and not misc.iterable(names)
                and not callable(names)):
            raise TypeError('`names` should be None, iterable or callable')

        super().__init__(**kwargs)

        self._names  = names
        self._colors = colors
        self._texts  = []

    def _configured(self, params, sinkParams):
        super()._configured(params, sinkParams)

        if misc.iterable(self._names) and len(self._names) != self._channels:
            raise ValueError('`names` should be the same size as number of'
                ' `channels`')

        if self._colors is not None and len(self._colors) != self._channels:
            raise ValueError('`colors` should be the same size as number of'
                ' `channels`')

        # init local vars that depend on node configuration (fs and channels)
        if self._colors == None:
            self._colors = _defaultColors[np.arange(self._channels)
                % len(_defaultColors)]
            # self._colors  = np.random.uniform(size=(self._channels, 3),
            #     low=.3, high=1)
        self._zooms   = int(np.round(np.log2(
                        self._fs/7500*self._figure.tsRange)))+1
        self._zoom    = self._zooms-1
        self._fss     = [None]*self._zooms
        self._indices = [None]*self._zooms
        self._buffers = [None]*self._zooms

        self._program.vert['channel_count'   ] = str(self._channels)
        self._program.frag['channel_count'   ] = str(self._channels)
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

            self._fss    [i] = fs
            self._indices[i] = np.arange(self._channels*nsRange).reshape(
                (self._channels,-1))
            self._buffers[i] = pipeline.CircularBuffer(
                duration=self._figure.tsRange)

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
            self._texts[i].transforms.configure(canvas=self._canvas,
                viewport=self._canvas.viewport)

            x = fig._pxlPos[0] + fig._pxlMargins[0] - 5
            y = (fig._pxlViewSize[1] * ((i+.5)/self._channels*self._untSize[1]
                - .5 + self._untPos[1]) * fig._zoom[1]
                - fig._pan[1]/2 * fig._pxlViewSize[1] + fig._pxlViewCenter[1])
            self._texts[i].pos = (x, y)
            self._texts[i].visible = (fig._pxlMargins[1] < y - fig._pxlPos[1]
                < fig._pxlMargins[3] + fig._pxlViewSize[1])

    def on_resize(self, event):
        self._updateTexts()

    def zoomed(self):
        if self._program is None: return

        self._updateTexts()

        zoom = self._zooms - 1 - np.log2(self._figure._zoom[0])
        zoom = int(np.round(zoom))
        zoom = np.clip(zoom, 0, self._zooms-1)

        if zoom != self._zoom:
            with self._lock:
                self._zoom = zoom
                self._updateProgram()

    def panned(self):
        if self._program is None: return

        self._updateTexts()

    def draw(self):
        super().draw('line_strip', smooth=True)
        for text in self._texts:
            text.draw()

    def _written(self, data, source):
        super()._written(data, source)
        with self._lock:
            buffer = self._buffers[self._zoom]
            self._program['u_ns'  ] = buffer.ns
            self._program['a_data'] = buffer.data.astype(np.float32)


class EpochPlot(Plot, pipeline.Node):
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

    @property
    def aux(self):
        return self._aux

    def __init__(self, name=None, color=(0,1,0,.6), **kwargs):
        '''
        Args:
            name:
            color:
        '''

        if not isinstance(kwargs.get('figure',None), Scope):
            raise TypeError('`figure` should be an instance of Scope')

        if name is not None and not isinstance(name, str):
            raise TypeError('`names should be None or a string`')

        super().__init__(**kwargs)

        self._color   = color
        self._name    = name
        self._cache   = 100*8    # cache 100 epochs
        self._pointer = 0
        self._partial = None
        self._index   = np.arange(self._cache, dtype=np.float32)
        self._data    = np.zeros (self._cache, dtype=np.float32)-1
        self._aux     = pipeline.Auxillary(
            self._auxConfigured, self._auxWritten)

        self._program.frag['rand'      ] = _glslRand

        self._program     ['u_color'   ] = self._color
        self._program     ['u_ts'      ] = 0
        self._program     ['u_ts_range'] = self._figure.tsRange
        self._program     ['u_ts_fade' ] = self._figure.tsFade

        self._program     ['a_index'   ] = self._index
        self._program     ['a_data'    ] = self._data
        self._program     ['a_data2'   ] = self._data.reshape(
                                                (-1,2))[:,::-1].ravel()

        if name is None:
            self._text = None
        else:
            self._text = vp.visuals.TextVisual(name, bold=True, font_size=10,
                rotation=0, color=self._color) #self._figure.fgColor)
            self._text.anchors = ('right', 'center')

        # self._updateText()

    def _updateText(self):
        if not self._text: return

        self._text.transforms.configure(canvas=self._canvas,
            viewport=self._canvas.viewport)

        fig = self._figure

        x = fig._pxlPos[0] + fig._pxlMargins[0] - 5
        y = (fig._pxlViewSize[1] * (self._untSize[1]/2
            - .5 + self._untPos[1]) * fig._zoom[1]
            - fig._pan[1]/2 * fig._pxlViewSize[1] + fig._pxlViewCenter[1])
        self._text.pos = (x, y)
        self._text.visible = (fig._pxlMargins[1] < y - fig._pxlPos[1]
            < fig._pxlMargins[3] + fig._pxlViewSize[1])

    def _auxConfigured(self):
        self._program['u_fs'] = self._aux.fs

    def _auxWritten(self):
        self._program['u_ts'] = self._aux.ts

    def on_resize(self, event):
        self._updateText()

    def zoomed(self):
        self._updateText()

    def panned(self):
        self._updateText()

    def draw(self):
        super().draw('triangle_strip')
        if self._text:
            self._text.draw()

    def _written(self, data, source):
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

        super()._written(data, source)


class SpikePlot(Plot, pipeline.Node):
    _vertShaderSource = '''
        uniform   vec2  u_plot_pos;       // unit in scene space
        uniform   vec2  u_plot_size;      // unit in scene space
        uniform   vec3  u_bg_color;
        uniform   vec3  u_color;
        uniform   int   u_spike_count;
        uniform   int   u_spike_index;
        uniform   int   u_spike_length;

        attribute float a_index;
        attribute float a_data;

        varying   float v_spike_index;
        varying   float v_sample_index;
        varying   vec2  v_pos_raw;
        varying   vec2  v_pos_plot;
        varying   vec2  v_pos_scene;
        varying   vec2  v_pos_view;
        varying   vec2  v_pos_figure;
        varying   vec2  v_pos_canvas;
        varying   vec3  v_color;

        void main() {
            v_spike_index  = floor(a_index / u_spike_length);
            v_sample_index = mod(a_index, u_spike_length);

            v_pos_raw      = vec2(v_sample_index, a_data);

            // all calculated coordinates below are in normalized space, i.e.:
            // -1 bottom-left to +1 top-right

            // channel
            v_pos_plot     = $transform(v_pos_raw,
                2. / (u_spike_length - 1), 1,
                -1, 0);

            // scene (all plots, before pan & zoom)
            v_pos_scene    = $transform(v_pos_plot,
                u_plot_size,
                ((u_plot_pos + u_plot_size/2) * 2 - 1) * vec2(1,-1));

            // view (after pan & zoom)
            v_pos_view     = $transform_view(v_pos_scene);

            // figure (title and axes)
            v_pos_figure   = $transform_figure(v_pos_view);

            // canvas
            v_pos_canvas   = $transform_canvas(v_pos_figure);

            gl_Position    = vec4(v_pos_canvas, 0, 1);
        }
        '''

    _fragShaderSource = '''
        uniform   vec2  u_plot_pos;       // unit in scene space
        uniform   vec2  u_plot_size;      // unit in scene space
        uniform   vec3  u_bg_color;
        uniform   vec3  u_color;
        uniform   int   u_spike_count;
        uniform   int   u_spike_index;
        uniform   int   u_spike_length;

        varying   float v_spike_index;
        varying   float v_sample_index;
        varying   vec2  v_pos_raw;
        varying   vec2  v_pos_plot;
        varying   vec2  v_pos_scene;
        varying   vec2  v_pos_view;
        varying   vec2  v_pos_figure;
        varying   vec2  v_pos_canvas;
        varying   vec3  v_color;

        void main() {
            //float ns = mod(u_ns, u_ns_range);
            u_bg_color;

            // discard the fragments between the channels
            // (emulate glMultiDrawArrays)
            if (0 < fract(v_spike_index))
                discard;

            // clip to view space
            if (1 < abs(v_pos_view.x) || 1 < abs(v_pos_view.y))
                discard;

            // discard unwritten samples in the first time window
            if (u_spike_index <= v_spike_index)
                discard;

            // calculate fading factor
            float fade = 1 - mod(u_spike_index-1-v_spike_index,
                u_spike_count) / u_spike_count;

            // set fragment color
            //gl_FragColor = vec4(u_color*fade + u_bg_color*(1-fade), 1.);
            gl_FragColor = vec4(u_color, fade);
        }
        '''

    @property
    def aux(self):
        return self._aux

    def __init__(self, name=None, color=(0,0,0), **kwargs):
        self._fs          = None
        self._spikeLength = None

        super().__init__(**kwargs)

        self._name       = name
        self._color      = color
        self._spikeCount = 20
        self._spikeIndex = 0
        self._aux        = pipeline.Auxillary(
            self._auxConfigured, self._auxWritten)

        self._program['u_color'      ] = self._color
        self._program['u_spike_count'] = self._spikeCount
        self._program['u_spike_index'] = self._spikeIndex

        if name is None:
            self._text = None
        else:
            self._text = vp.visuals.TextVisual(name, bold=True, font_size=14,
                color=color, anchor_x='left', anchor_y='top')

    def _configuring(self, params, sinkParams):
        super()._configuring(params, sinkParams)

        if 'spikeLength' not in params:
            raise ValueError('SpikePlot requires `spikeLength`')

        if 'channels' in params and params['channels'] != 1:
            raise ValueError('SpikePlot can only plot a single `channel`')

    def _configured(self, params, sinkParams):
        super()._configured(params, sinkParams)

        self._spikeLength = params['spikeLength']
        self._index       = np.arange(self._spikeCount*self._spikeLength,
            dtype=np.float32)
        self._data        = np.zeros((self._spikeCount, self._spikeLength),
            dtype=np.float32)

        self._program['u_spike_length'] = self._spikeLength
        self._program['a_index'       ] = self._index
        self._program['a_data'        ] = self._data

    def _updateText(self):
        if not self._text: return

        self._text.transforms.configure(canvas=self._canvas,
            viewport=self._canvas.viewport)

        fig = self._figure

        self._text.pos = fig._pxlViewCenter - fig._pxlViewSize/2 + 5

    def _auxConfigured(self):
        pass

    def _auxWritten(self):
        # self._program['u_ts'] = self._aux.ts
        pass

    def _written(self, data, source):
        for spike in data:
            if len(spike) != self._spikeLength:
                raise ValueError('Spike length discrepancy (%d != %d)' %
                    (len(spike), self._spikeLength))
            spikeIndex = self._spikeIndex % self._spikeCount
            self._data[spikeIndex] = spike
            self._spikeIndex += 1

        with self._lock:
            self._program['u_spike_index'] = self._spikeIndex
            self._program['a_data'       ] = self._data

        super()._written(data, source)

    def on_resize(self, event):
        self._updateText()

    def draw(self):
        super().draw('line_strip', smooth=True)
        # if self._text:
        #     self._text.draw()


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
        super(MainWindow, self).__init__()
        self.setWindowTitle('Graphics Acceleration Demo')

        self.canvas     = Canvas()
        self.scope      = Scope(parent=self.canvas, untPos=(0,0),
            untSize=(.7,1), pxlMargins=(60,15,15,15))
        self.analogPlot = AnalogPlot(figure=self.scope, untPos=(0,0),
            untSize=(1,.9), names=str)
        self.targetPlot = EpochPlot(figure=self.scope, untPos=(0,.9),
            untSize=(1,.05), name='Target', color=(0,1,0,.7))
        self.pumpPlot   = EpochPlot(figure=self.scope, untPos=(0,.95),
            untSize=(1,.05), name='Pump', color=(0,0,1,.7))
        # self.targetPlot.write([2,4])

        self.spikeCont = Container(parent=self.canvas, untPos=(.7,0),
            untSize=(.3,1), pxlMargins=(0,15,15,15))

        self.channels = 16
        self.spikeFigures = [None]*self.channels
        self.spikePlots   = [None]*self.channels
        for i in range(self.channels):
            self.spikeFigures[i] = Figure(parent=self.spikeCont,
                untPos=(.25*(i%4),.25*(i//4)), untSize=(.25,.25),
                pxlMargins=(0,0,0 if i%4==3 else 1,0 if i//4==3 else 1))
                # untPos=(.5*(i%2),.125*(i//2)), untSize=(.5,.125),
                # pxlMargins=(0,0,0 if i%2==1 else 1,0 if i//2==7 else 1))
            self.spikePlots[i] = SpikePlot(figure=self.spikeFigures[i],
                name=str(i+1), color=_defaultColors[i%len(_defaultColors)])

        self.generator = SpikeGenerator(fs=31.25e3, channels=self.channels)
        self.filter    = pipeline.LFilter(fl=100, fh=6000)
        self.grandAvg  = pipeline.GrandAverage()

        # setup pipeline
        self.generator >> self.grandAvg >> self.filter >> (
            self.scope,
            self.analogPlot,
            self.targetPlot.aux,
            self.pumpPlot.aux,
            SpikeDetector() >> pipeline.Split() >> self.spikePlots
            )
        self.generator.start()

        self.canvas.keyReleased = self.keyReleased

        self.button = QtWidgets.QPushButton('Test')

        mainLayout = QtWidgets.QHBoxLayout()
        mainLayout.setContentsMargins(0, 0, 0, 0)
        mainLayout.addWidget(self.canvas.native)
        # mainLayout.addWidget(self.button)

        self.setLayout(mainLayout)

    def keyReleased(self, event):
        if event.key == 'space':
            if self.generator.paused:
                self.generator.start()
            else:
                self.generator.pause()
        elif event.key == 't':
            self.targetPlot.write(self.generator.ts)
        elif event.key == 'p':
            self.pumpPlot.write(self.generator.ts)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
