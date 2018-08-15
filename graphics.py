'''GPU accelerated graphics and plotting using vispy.


This file is part of the EARS project: https://github.com/nalamat/ears
Copyright (C) 2017-2018 Nima Alamatsaz <nima.alamatsaz@njit.edu>
Copyright (C) 2017-2018 Antje Ihlefeld <antje.ihlefeld@njit.edu>

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
'''

import math
import numpy      as     np
import datetime   as     dt
# from   lttb       import downsample
from   vispy      import app, gloo, visuals
from   PyQt5      import QtCore, QtGui, QtWidgets
from   OpenGL.GL  import *
# from   OpenGL.GLU import *
# from border import BorderVisual

_vertShaderSource = '''
//#version 450

uniform   float u_line_count;
uniform   vec3  u_bg_color;
uniform   vec4  u_axes_size;      // pixels
uniform   vec2  u_figure_size;    // pixels
uniform   vec2  u_view_size;      // pixels
uniform   vec2  u_view_center;    // pixels
uniform   vec2  u_zoom;
uniform   vec2  u_pan;            // pixels
uniform   float u_ns;
uniform   float u_ns_range;
uniform   float u_ns_fade;

attribute float a_line_index;
attribute float a_time_index;     // sample number
attribute float a_data;
attribute vec3  a_color;

varying   float v_line_index;
varying   float v_time_index;
varying   vec2  v_pos_raw;
varying   vec2  v_pos_signal;
varying   vec2  v_pos_plot;
varying   vec2  v_pos_view;
varying   vec2  v_pos_figure;
varying   vec3  v_color;

vec2 transform(vec2 p, vec2 a, vec2 b) {
    return a*p + b;
}

vec2 transform(vec2 p, float ax, float ay, float bx, float by) {
    return transform(p, vec2(ax, ay), vec2(bx, by));
}

void main() {
    v_pos_raw = vec2(a_time_index, a_data);

    // all calculated coordinates below are in normalized space, i.e.:
    // -1 bottom-left to +1 top-right

    // signal
    v_pos_signal = transform(v_pos_raw,
        2/(u_ns_range-1), 1,
        -1, 0);

    // plot
    v_pos_plot = transform(v_pos_signal,
        1, 1/u_line_count,
        0, 1 - 2*(a_line_index+.5)/u_line_count);

    // view (pan & zoom)
    v_pos_view = transform(v_pos_plot,
        u_zoom,
        u_pan);

    // figure (title and axes)
    v_pos_figure = transform(v_pos_view,
        u_view_size / u_figure_size,
        (u_view_center * 2 / u_figure_size - 1) * vec2(1,-1));
        //(u_axes_size.xw - u_axes_size.zy) / u_figure_size);

    gl_Position  = vec4(v_pos_figure, 0, 1);

    // pipe to fragment shader
    v_line_index = a_line_index;
    v_time_index = a_time_index;
    v_color      = a_color;
}
'''

_fragShaderSource = '''
//#version 450

uniform vec3  u_bg_color;
uniform vec2  u_view_size;      // pixels
uniform vec2  u_zoom;
uniform float u_ns;
uniform float u_ns_range;
uniform float u_ns_fade;

varying float v_time_index;
varying float v_line_index;
varying vec2  v_pos_raw;
varying vec2  v_pos_signal;
varying vec2  v_pos_plot;
varying vec2  v_pos_view;
varying vec2  v_pos_figure;
varying vec3  v_color;

float fade_lin(float ns) {
    return (v_time_index-ns)/u_ns_fade;
}

float fade_sin(float ns) {
    return sin(((v_time_index-ns)/u_ns_fade*2-1)*3.1415/2)/2+.5;
}

float fade_pow3(float ns) {
    return pow((v_time_index-ns)/u_ns_fade*2-1, 3)/2+.5;
}

void main() {
    // discard the fragments between the signals, emulate glMultiDrawArrays
    if (0 < fract(v_line_index))
        discard;

    // clip to view space
    if (1 < abs(v_pos_view.x) || 1 < abs(v_pos_view.y))
        discard;

    //if (u_ns <= v_time_index)
        //discard;

    // calculate alpha
    float alpha = 1.;
    float ns = mod(u_ns, u_ns_range);
    if (ns <= v_time_index && v_time_index <= ns+u_ns_fade)
        alpha = fade_sin(ns);
    if (ns-u_ns_range <= v_time_index &&
            v_time_index <= ns+u_ns_fade-u_ns_range)
        alpha = fade_sin(ns-u_ns_range);

    // set fragment color
    gl_FragColor = vec4(v_color*alpha + u_bg_color*(1-alpha), 1.);
    //gl_FragColor = vec4(v_color, alpha);
}
'''

class AnalogPlot:
    def __init__(self, fs=5e3, fs2=1e3, lineCount=16, tsRange=10, tsFade=10,
            bgColor=(0,0,.2), fgColor=(0,1,1)):
        pass

class Canvas(app.Canvas):
    @property
    def zoom(self):
        return self._zoom

    @zoom.setter
    def zoom(self, zoom):
        zoom = np.array(zoom)
        zoom = zoom.clip(1, 1000)
        self._program['u_zoom'] = zoom
        self._zoom = zoom
        self._updateAxes()
        self.update()

    @property
    def pan(self):
        return self._pan

    @pan.setter
    def pan(self, pan):
        # `pan` is in normalized view coordinates
        panLimits = self._zoom-1
        pan = np.array(pan)
        pan = pan.clip(-panLimits, panLimits)
        self._program['u_pan'] = pan
        self._pan = pan
        self._updateAxes()
        self.update()

    def __init__(self, fs=10e3, fs2=1e3, lineCount=16, tsRange=10, tsFade=10,
        bgColor=(0,0,.2), fgColor=(0,1,1)):

        super().__init__(title='Use your wheel to zoom!', keys='interactive',
            vsync=True, config=dict(samples=4))

        self._fs        = fs
        self._fs2       = fs2
        self._lineCount = lineCount
        self._tsRange   = tsRange
        self._tsFade    = tsFade
        self._bgColor   = bgColor
        self._fgColor   = fgColor

        self._axesSize  = np.array([30, 15, 15, 15])
        self._zoom      = np.array([1, 1])
        self._pan       = np.array([0, 0])

        self._ns        = 0
        self._nsRange   = int(tsRange*fs)
        self._nsFade    = int(tsFade *fs)

        # random sines
        time   = np.tile(np.arange(self._nsRange)/fs,
            (lineCount,1))
        phases = np.repeat(np.random.uniform(size=(lineCount,1),
            low=0, high=np.pi), self._nsRange, axis=1)
        freqs  = np.repeat(np.random.uniform(size=(lineCount,1),
            low=1, high=5), self._nsRange, axis=1)
        amps   = np.repeat(np.random.uniform(size=(lineCount,1),
            low=.05, high=.5), self._nsRange, axis=1)
        self._sines = amps * np.sin(2*np.pi*freqs*time+phases)

        # normal noise
        amps2  = (.05 + .15 * np.random.rand(lineCount, 1)) * 1
        self._noise = lambda ns: amps2 * np.random.randn(lineCount, ns)

        # Line and time index of each vertex
        self._lineIndex = np.repeat(np.arange(lineCount), self._nsRange)
        self._timeIndex = np.tile(np.arange(self._nsRange), lineCount)

        # Generate the signals as a (lineCount, nsRange) array.
        self._data = self._noise(self._nsRange) + self._sines

        # Color of each vertex (TODO: make it more efficient by using a
        # GLSL-based color map and the index).
        self._color = np.repeat(np.random.uniform(size=(lineCount, 3),
            low=.3, high=1), self._nsRange, axis=0)
        # self._color2 = np.random.uniform(size=(lineCount, 3), low=.3, high=1)

        gloo.set_state(clear_color=self._bgColor, blend=True, depth_test=False)

        self._program = gloo.Program(_vertShaderSource, _fragShaderSource)
        self._program['u_line_count' ] = lineCount
        self._program['u_bg_color'   ] = bgColor
        self._program['u_axes_size'  ] = self._axesSize
        self._program['u_zoom'       ] = self._zoom
        self._program['u_pan'        ] = self._pan
        self._program['u_ns'         ] = self._ns
        self._program['u_ns_range'   ] = self._nsRange
        self._program['u_ns_fade'    ] = self._nsFade
        self._program['a_line_index' ] = self._lineIndex.astype(np.float32)
        self._program['a_time_index' ] = self._timeIndex.astype(np.float32)
        self._program['a_data'       ] = self._data.astype(np.float32)
        self._program['a_color'      ] = self._color.astype(np.float32)

        self._yTicksText = [None]*lineCount
        for i in range(lineCount):
            text = visuals.TextVisual(str(i), bold=True, color=fgColor,
                font_size=14)
            text.anchors = ('right', 'center')
            self._yTicksText[i] = text
        self._fpsText = visuals.TextVisual('0.0', bold=True, color=fgColor,
            pos=(1,1), font_size=8)
        self._fpsText.anchors = ('left', 'top')
        self._mouseText = visuals.TextVisual('', bold=True, color=fgColor,
            pos=(30,1), font_size=8)
        self._mouseText.anchors = ('left', 'top')
        self._border = visuals._BorderVisual(pos=(0,0), halfdim=(0,0),
            border_width=1, border_color=fgColor)
        self._line = visuals.LineVisual(np.array([[0, 0], [500,500]]))

        self._visuals = visuals.CompoundVisual(
            self._yTicksText +
            [self._fpsText,
            self._mouseText,
            self._border,
            # self._line,
            ])

        self._updateLast  = None
        self._updateTimes = []    # keep last update times for measuring FPS
        self._startTime   = dt.datetime.now()
        self._timer       = app.Timer('auto', connect=self.on_timer)

        self._timer.start()

    def _view2norm(self, p):
        return p * 2 / self._viewSize * [1, -1]

    def _norm2view(self, p):
        return p / 2 * self._viewSize * [1, -1]

    def _updateAxes(self):
        for i in range(self._lineCount):
            x = self._axesSize[0]-5
            y = ((self._viewSize[1]/self._lineCount*(i+.5)
                - self._viewCenter[1] + self._axesSize[1]) * self._zoom[1]
                - self._pan[1]/2*self._viewSize[1] + self._viewCenter[1])
            self._yTicksText[i].pos = (x, y)
            self._yTicksText[i].visible = (self._axesSize[1] < y
                < self._axesSize[1] + self._viewSize[1])

    def on_resize(self, event):
        vp = (0, 0, *self.physical_size)
        self.context.set_viewport(*vp)
        self._visuals.transforms.configure(canvas=self, viewport=vp)
        self._line.transforms.configure(canvas=self, viewport=vp)

        # all coordinates below are pixels in document space, i.e.:
        # 0 top-left to window size bottom-right
        self._figureSize = np.array(self.size)
        self._viewCenter = (self._figureSize + self._axesSize[0:2] -
            self._axesSize[2:4])/2
        self._viewSize   = (self._figureSize - self._axesSize[0:2] -
            self._axesSize[2:4])

        self._program['u_figure_size'] = self._figureSize
        self._program['u_view_size'  ] = self._viewSize
        self._program['u_view_center'] = self._viewCenter

        self._updateAxes()

        self._border.pos     = self._viewCenter
        self._border.halfdim = self._viewSize/2

    def on_mouse_wheel(self, event):
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
        delta = self._view2norm(event.pos - self._viewCenter)
        self.pan = delta * (1 - scale) + self.pan * scale

    def on_mouse_press(self, event):
        if (event.button == 1 and 'Control' in event.modifiers):
            # start pan
            event.pan = self.pan

    def on_mouse_move(self, event):
        if (event.press_event and event.press_event.button == 1 and
                'Control' in event.press_event.modifiers):
            # pan
            delta = self._view2norm(event.pos-event.press_event.pos)
            self.pan = event.press_event.pan + delta
        self._mouseText.text = str(event.pos)

    def on_mouse_release(self, event):
        if (event.press_event and event.press_event.button == 1 and
                'Control' in event.press_event.modifiers):
            # stop pan
            delta = self._view2norm(event.pos-event.press_event.pos)
            self.pan = event.press_event.pan + delta

    def on_key_release(self, event):
        if 'Control' in event.modifiers and event.key == '0':
            self.zoom = [1, 1]
            self.pan  = [0, 0]
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
        '''Add some data at the end of each signal (real-time signals).'''
        # if self._last:
        ns = int((dt.datetime.now()-self._startTime).total_seconds()*self._fs)
        mask = np.arange(self._ns,ns) % self._nsRange
        self._data[:,mask] = self._noise(ns-self._ns) + self._sines[:,mask]
        # self._program['a_data'] = self._data.astype(np.float32).repeat(2)
        self._program['u_ns'] = ns
        self._ns = ns
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

        gloo.clear()
        # glEnable(GL_LINE_STIPPLE)
        # glLineStipple(50, 0b1010101010101010)
        # self._line.draw()
        # glDisable(GL_LINE_STIPPLE)
        glLineWidth(1)
        self._program.draw('line_strip')
        self._visuals.draw()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        # self.plot = RealTimePlotWidget(self)
        self.plot = Canvas()
        # self.plot.native.format().setSamples(16)
        self.button = QtWidgets.QPushButton('Test')

        mainLayout = QtWidgets.QHBoxLayout()
        mainLayout.setContentsMargins(0, 0, 0, 0)
        # mainLayout.addWidget(self.plot)
        mainLayout.addWidget(self.plot.native)
        # mainLayout.addWidget(self.button)

        self.setLayout(mainLayout)


if __name__ == '__main__':

    # c = Canvas()
    # c.show()
    # app.run()

    # surfaceFormat = QtGui.QSurfaceFormat()
    # surfaceFormat.setSamples(16)
    # surfaceFormat.setProfile(QtGui.QSurfaceFormat.CoreProfile)
    # surfaceFormat.setMajorVersion(4)
    # surfaceFormat.setMinorVersion(1)
    # QtGui.QSurfaceFormat.setDefaultFormat(surfaceFormat)

    qtapp = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    qtapp.exec_()
