'''Real-time plotting using pyqtgraph and PyQt5 packages.


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

import time
import logging
import threading
import scipy.signal
import numpy        as     np
import scipy        as     sp
import tables       as     tb
import datetime     as     dt
import pyqtgraph    as     pg
from   PyQt5        import QtCore, QtWidgets, QtGui

import misc
import hdf5


log = logging.getLogger(__name__)
filters = tb.Filters(complevel=9, ) #complib='blosc')


# imports for Axis
from pyqtgraph.python2_3 import asUnicode
from pyqtgraph.Point import Point
from pyqtgraph import debug as debug
import weakref
from pyqtgraph import functions as fn
from pyqtgraph import getConfigOption
from pyqtgraph.graphicsItems.GraphicsWidget import GraphicsWidget

class Axis(pg.AxisItem):
    def generateDrawSpecs(self, p):
        """
        Calls tickValues() and tickStrings() to determine where and how ticks should
        be drawn, then generates from this a set of drawing commands to be
        interpreted by drawPicture().
        """
        profiler = debug.Profiler()

        #bounds = self.boundingRect()
        bounds = self.mapRectFromParent(self.geometry())

        linkedView = self.linkedView()
        if linkedView is None or self.grid is False:
            tickBounds = bounds
        else:
            tickBounds = linkedView.mapRectToItem(self, linkedView.boundingRect())

        if self.orientation == 'left':
            span = (bounds.topRight(), bounds.bottomRight())
            tickStart = tickBounds.right()
            tickStop = bounds.right()
            tickDir = -1
            axis = 0
        elif self.orientation == 'right':
            span = (bounds.topLeft(), bounds.bottomLeft())
            tickStart = tickBounds.left()
            tickStop = bounds.left()
            tickDir = 1
            axis = 0
        elif self.orientation == 'top':
            span = (bounds.bottomLeft(), bounds.bottomRight())
            tickStart = tickBounds.bottom()
            tickStop = bounds.bottom()
            tickDir = -1
            axis = 1
        elif self.orientation == 'bottom':
            span = (bounds.topLeft(), bounds.topRight())
            tickStart = tickBounds.top()
            tickStop = bounds.top()
            tickDir = 1
            axis = 1
        #print tickStart, tickStop, span

        ## determine size of this item in pixels
        points = list(map(self.mapToDevice, span))
        if None in points:
            return
        lengthInPixels = Point(points[1] - points[0]).length()
        if lengthInPixels == 0:
            return

        # Determine major / minor / subminor axis ticks
        if self._tickLevels is None:
            tickLevels = self.tickValues(self.range[0], self.range[1], lengthInPixels)
            tickStrings = None
        else:
            ## parse self.tickLevels into the formats returned by tickLevels() and tickStrings()
            tickLevels = []
            tickStrings = []
            for level in self._tickLevels:
                values = []
                strings = []
                tickLevels.append((None, values))
                tickStrings.append(strings)
                for val, strn in level:
                    values.append(val)
                    strings.append(strn)

        ## determine mapping between tick values and local coordinates
        dif = self.range[1] - self.range[0]
        if dif == 0:
            xScale = 1
            offset = 0
        else:
            if axis == 0:
                xScale = -bounds.height() / dif
                offset = self.range[0] * xScale - bounds.height()
            else:
                xScale = bounds.width() / dif
                offset = self.range[0] * xScale

        xRange = [x * xScale - offset for x in self.range]
        xMin = min(xRange)
        xMax = max(xRange)

        profiler('init')

        tickPositions = [] # remembers positions of previously drawn ticks

        ## compute coordinates to draw ticks
        ## draw three different intervals, long ticks first
        tickSpecs = []
        for i in range(len(tickLevels)):
            tickPositions.append([])
            ticks = tickLevels[i][1]

            ## length of tick
            tickLength = self.style['tickLength'] / ((i*0.5)+1.0)

            lineAlpha = 255 / (i+1)
            if self.grid is not False:
                lineAlpha *= self.grid/255. * np.clip((0.05  * lengthInPixels / (len(ticks)+1)), 0., 1.)

            for v in ticks:
                ## determine actual position to draw this tick
                x = (v * xScale) - offset
                if x < xMin or x > xMax:  ## last check to make sure no out-of-bounds ticks are drawn
                    tickPositions[i].append(None)
                    continue
                tickPositions[i].append(x)

                p1 = [x, x]
                p2 = [x, x]
                p1[axis] = tickStart
                p2[axis] = tickStop
                if self.grid is False:
                    p2[axis] += tickLength*tickDir
                tickPen = self.pen()
                color = tickPen.color()
                color.setAlpha(lineAlpha)
                tickPen.setColor(color)
                tickSpecs.append((tickPen, Point(p1), Point(p2)))
        profiler('compute ticks')


        if self.style['stopAxisAtTick'][0] is True:
            stop = max(span[0].y(), min(map(min, tickPositions)))
            if axis == 0:
                span[0].setY(stop)
            else:
                span[0].setX(stop)
        if self.style['stopAxisAtTick'][1] is True:
            stop = min(span[1].y(), max(map(max, tickPositions)))
            if axis == 0:
                span[1].setY(stop)
            else:
                span[1].setX(stop)
        axisSpec = (self.pen(), span[0], span[1])


        textOffset = self.style['tickTextOffset'][axis]  ## spacing between axis and text
        #if self.style['autoExpandTextSpace'] is True:
            #textWidth = self.textWidth
            #textHeight = self.textHeight
        #else:
            #textWidth = self.style['tickTextWidth'] ## space allocated for horizontal text
            #textHeight = self.style['tickTextHeight'] ## space allocated for horizontal text

        textSize2 = 0
        textRects = []
        textSpecs = []  ## list of draw

        # If values are hidden, return early
        if not self.style['showValues']:
            return (axisSpec, tickSpecs, textSpecs)

        for i in range(min(len(tickLevels), self.style['maxTextLevel']+1)):
            ## Get the list of strings to display for this level
            if tickStrings is None:
                spacing, values = tickLevels[i]
                strings = self.tickStrings(values, self.autoSIPrefixScale * self.scale, spacing)
            else:
                strings = tickStrings[i]

            if len(strings) == 0:
                continue

            ## ignore strings belonging to ticks that were previously ignored
            for j in range(len(strings)):
                if tickPositions[i][j] is None:
                    strings[j] = None

            ## Measure density of text; decide whether to draw this level
            rects = []
            for s in strings:
                if s is None:
                    rects.append(None)
                else:
                    br = p.boundingRect(QtCore.QRectF(0, 0, 100, 100), QtCore.Qt.AlignCenter, asUnicode(s))
                    ## boundingRect is usually just a bit too large
                    ## (but this probably depends on per-font metrics?)
                    br.setHeight(br.height() * 0.8)

                    rects.append(br)
                    textRects.append(rects[-1])

            if len(textRects) > 0:
                ## measure all text, make sure there's enough room
                if axis == 0:
                    textSize = np.sum([r.height() for r in textRects])
                    textSize2 = np.max([r.width() for r in textRects])
                else:
                    textSize = np.sum([r.width() for r in textRects])
                    textSize2 = np.max([r.height() for r in textRects])
            else:
                textSize = 0
                textSize2 = 0

            if i > 0:  ## always draw top level
                ## If the strings are too crowded, stop drawing text now.
                ## We use three different crowding limits based on the number
                ## of texts drawn so far.
                textFillRatio = float(textSize) / lengthInPixels
                finished = False
                for nTexts, limit in self.style['textFillLimits']:
                    if len(textSpecs) >= nTexts and textFillRatio >= limit:
                        finished = True
                        break
                if finished:
                    break

            #spacing, values = tickLevels[best]
            #strings = self.tickStrings(values, self.scale, spacing)
            # Determine exactly where tick text should be drawn
            for j in range(len(strings)):
                vstr = strings[j]
                if vstr is None: ## this tick was ignored because it is out of bounds
                    continue
                vstr = asUnicode(vstr)
                x = tickPositions[i][j]
                #textRect = p.boundingRect(QtCore.QRectF(0, 0, 100, 100), QtCore.Qt.AlignCenter, vstr)
                textRect = rects[j]
                height = textRect.height()
                width = textRect.width()
                #self.textHeight = height
                offset = max(0,self.style['tickLength']) + textOffset
                if self.orientation == 'left':
                    textFlags = QtCore.Qt.TextDontClip|QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter
                    rect = QtCore.QRectF(tickStop-offset-width, x-(height/2), width, height)
                elif self.orientation == 'right':
                    textFlags = QtCore.Qt.TextDontClip|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter
                    rect = QtCore.QRectF(tickStop+offset, x-(height/2), width, height)
                elif self.orientation == 'top':
                    textFlags = QtCore.Qt.TextDontClip|QtCore.Qt.AlignCenter|QtCore.Qt.AlignBottom
                    rect = QtCore.QRectF(x-width/2., tickStop-offset-height, width, height)
                elif self.orientation == 'bottom':
                    textFlags = QtCore.Qt.TextDontClip|QtCore.Qt.AlignCenter|QtCore.Qt.AlignTop
                    rect = QtCore.QRectF(x-width/2., tickStop+offset, width, height)

                #p.setPen(self.pen())
                #p.drawText(rect, textFlags, vstr)
                textSpecs.append((rect, textFlags, vstr))
        profiler('compute text')

        ## update max text size if needed.
        self._updateMaxTextSize(textSize2)

        return (axisSpec, tickSpecs, textSpecs)

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        profiler = debug.Profiler()

        p.setRenderHint(p.Antialiasing, False)
        p.setRenderHint(p.TextAntialiasing, True)

        ## draw long line along axis
        pen, p1, p2 = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)
        p.translate(0.5,0)  ## resolves some damn pixel ambiguity

        ## draw ticks
        for pen, p1, p2 in tickSpecs:
            p.setPen(pen)
            p.drawLine(p1, p2)
        profiler('draw ticks')

        ## Draw all text
        if self.tickFont is not None:
            p.setFont(self.tickFont)
        p.setPen(self.pen())
        # p.rotate(-90)
        for rect, flags, text in textSpecs:
            # r = QtCore.QRectF(-rect.y(), rect.x()+rect.width()-rect.height(), rect.height(), 0)
            p.drawText(rect, flags, text)
            #p.drawRect(rect)
        # p.rotate(90)
        profiler('draw text')


class ChannelPlotWidget(pg.PlotWidget):
    '''A Qt widget for managing and plotting multiple channels in real-time.'''

    # target frame rate per second
    @property
    def fps(self):
        return self._fps

    # actual frame rate per second
    @property
    def calculatedFPS(self):
        return self._calculatedFPS

    # time (x-axis) range in the plot window
    @property
    def xRange(self):
        return self._xRange

    # current timestamp (from latest sample in timeBase)
    @property
    def ts(self):
        if self._timeBase is None:
            raise AttributeError('No timeBase has been set')
        return self._timeBase.ts

    # min time in the current plot window
    # @property
    # def tsMin(self):
    #     return int(self.ts//self.tsRange*self.tsRange)

    def __init__(self, yLimits=(-1,1), yGrid=1, xRange=10, xGrid=1, fps=60,
            *args, **kwargs):
        '''
        Args:
            yLimits (tuple of float): Minimum and maximum values shown on the
                y-axis. Defaults to (1,-1).
            yGrid (float or list of float): Spacing between the horizontal grid
                lines if a single value is given, or the y at which horizontal
                grid lines are drawn if a list of values are given.
                Defaults to 1.
            xRange (float): Defaults to 10.
            xGrid (float): Spacing between vertical grid lines. Defaults to 1.
        '''

        super().__init__(*args, **kwargs, axisItems={'left':Axis('left')})

        # should not change after __init__
        self._yLimits     = yLimits
        self._yGrid       = yGrid
        self._xRange      = xRange
        self._yGrid       = yGrid
        self._fps         = fps
        self._calculatedFPS = 0

        self._channels    = []
        self._timeBase    = None
        self._updateTimes = []    # hold last update times for FPS calculation
        self._tsMinLast   = None

        self._timer       = QtCore.QTimer()
        self._timer.timeout.connect(self.updatePlot)
        self._timer.setInterval(1000/self.fps)

        # self.updateFPS    = None
        # self.updateTS     = None

        self.setBackground('w')
        self.setMouseEnabled(False, False)
        self.setMenuEnabled(False)
        self.setClipToView(False)
        self.disableAutoRange()
        self.setRange(xRange=(0, xRange), yRange=yLimits, padding=0)
        self.setLimits(xMin=0, xMax=xRange, yMin=yLimits[0], yMax=yLimits[1])
        self.hideButtons()

        # draw all axis as black (draw box)
        self.getAxis('left'  ).setPen(pg.mkPen('k'))
        self.getAxis('bottom').setPen(pg.mkPen('k'))
        self.getAxis('right' ).setPen(pg.mkPen('k'))
        self.getAxis('top'   ).setPen(pg.mkPen('k'))
        self.getAxis('bottom').setTicks([])
        self.getAxis('right' ).setTicks([])
        self.getAxis('right' ).show()
        self.getAxis('top'   ).show()
        self.getAxis('top'   ).setLabel('<br />Time (min:sec)')
        xticks = [(t, '%02d:%02d' % (t//60,t%60)) for t in range(xRange+1)]
        self.getAxis('top').setTicks([xticks])

        # draw vertical grid
        self._xGridMajor = [None] * int(np.ceil(xRange/xGrid)-1)
        self._xGridMinor = [None] * int(np.ceil(xRange/xGrid))
        for i in range(len(self._xGridMajor)):
            line = pg.InfiniteLine(i+1, 90, pen=pg.mkPen((150,150,150),
                style=QtCore.Qt.DashLine, cosmetic=True))
            self._xGridMajor[i] = line
            self.addItem(line)
        for i in range(len(self._xGridMinor)):
            line = pg.InfiniteLine(i+.5, 90, pen=pg.mkPen((150,150,150),
                style=QtCore.Qt.DotLine, cosmetic=True))
            self._xGridMinor[i] = line
            self.addItem(line)

        # draw horizontal grid
        if yGrid is None: yGridAt = []
        elif isinstance(yGrid, list): yGridAt = yGrid
        else: yGridAt = range(yLimits[0]+yGrid, yLimits[1], yGrid)
        for y in yGridAt:
            line = pg.InfiniteLine(y, 0, pen=pg.mkPen((150,150,150),
                style=QtCore.Qt.DashLine, cosmetic=True))
            self.addItem(line)

        yMajorTicks = [(y, '') for y in yGridAt]
        yMinorTicks = []
        self.getAxis('left').setTicks([yMajorTicks, yMinorTicks])
        self.getAxis('left').setStyle(tickLength=0)

    def addChannel(self, channel, label=None, labelOffset=0, timeBase=False):
        if not isinstance(channel, BaseChannel):
            raise TypeError('Channel should be an instance of BaseChannel'
                'or one of its subclasses')
        if channel in self._channels:
            raise RuntimeError('Cannot add a channel twice')

        self._channels.append(channel)
        if label:
            ticks = self.getAxis('left')._tickLevels
            ticks = [ticks[0]+[(labelOffset, ' ' + label)], ticks[1]]
            self.getAxis('left').setTicks(ticks)
        if timeBase:
            self._timeBase = channel
        # if not self._timer.isActive():
        #     self._updateLast = dt.datetime.now()
        #     self._timer.start()

    def start(self):
        if not self._timer.isActive():
            self._updateLast = dt.datetime.now()
            self._timer.start()

    def stop(self):
        if self._timer.isActive():
            self._timer.stop()
        self.updatePlot()

    def updatePlot(self):
        try:
            # calculate actual fps
            updateTime = (dt.datetime.now()-self._updateLast).total_seconds()
            self._updateTimes.append(updateTime)
            if len(self._updateTimes)>60: self._updateTimes.pop(0)
            updateMean = np.mean(self._updateTimes)
            fps = 1/updateMean if updateMean else 0
            self._calculatedFPS = fps
            self._updateLast = dt.datetime.now()

            ts         = None
            tsMin      = None
            nextWindow = False

            if self._timeBase is not None:
                ts = self._timeBase.ts
                tsMin = int(ts//self.xRange*self.xRange)

                # determine when plot advances to next time window
                if tsMin != self._tsMinLast:
                    self.setLimits(xMin=tsMin, xMax=tsMin+self.xRange)

                    # update vertical grid
                    ticks = [(t, '%02d:%02d' % (t//60,t%60)) for t in
                        range(tsMin, tsMin+self.xRange+1)]
                    self.getAxis('top').setTicks([ticks])
                    for i in range(len(self._xGridMajor)):
                        self._xGridMajor[i].setValue(tsMin+i+1)
                    for i in range(len(self._xGridMinor)):
                        self._xGridMinor[i].setValue(tsMin+i+.5)

                    log.debug('Advancing to next plot window at %d', tsMin)
                    self._tsMinLast = tsMin
                    nextWindow = True

            # if self.updateFPS is not None: self.updateFPS('%.1f' % fps)
            # if self.updateTS  is not None: self.updateTS ('%.2f' % ts )

            # update all channels
            for channel in self._channels:
                channel.updatePlot(ts, tsMin, nextWindow)

        except:
            log.exception('Failed to update plot, stopping plot timer')
            self._timer.stop()


class BaseChannel():
    '''Base class for all channels.

    Contains no functional code, only for class hierarchy purposes.
    '''
    pass


class AnalogChannel(BaseChannel):
    '''All-in-one class for storage and plotting of multiline analog signals.

    Utilizes the `data` module for storage in HDF5 file format.

    Needs to be paired with a ChannelPlotWidget for plotting.
    '''

    @property
    def lineCount(self):
        return self._lineCount

    @property
    def fs(self):
        '''Sampling frequency'''
        return self._fs

    @property
    def ns(self):
        '''Total number of samples (per line) added to channel'''
        # return self._ns
        return self._buffer1.nsWritten

    @property
    def ts(self):
        '''Last timestamp in seconds'''
        return self.ns/self._fs

    @property
    def plotWidget(self):
        return self._plotWidget

    @property
    def yScale(self):
        return self._yScale

    @yScale.setter
    def yScale(self, value):
        self._yScale = value
        if self._init: self.refresh()

    @property
    def flFilter(self):
        return self._flFilter

    @flFilter.setter
    def flFilter(self, value):
        self._flFilter = value

        with self._refreshLock:
            self._refreshFilter()
            if self._init: self._refresh()

    @property
    def fhFilter(self):
        return self._fhFilter

    @fhFilter.setter
    def fhFilter(self, value):
        self._fhFilter = value

        with self._refreshLock:
            self._refreshFilter()
            if self._init: self._refresh()

    @property
    def fsPlot(self):
        return self._fsPlot

    @fsPlot.setter
    def fsPlot(self, value):
        with self._refreshLock:
            if value and value != self.fs:
                # downsampling factor, i.e. number of consecutive samples to
                # average, must be integer
                self._dsFactor = round(self.fs/value)
                # recalculate the plotting sampling frequency
                value = self.fs / self._dsFactor
            else:
                self._dsFactor = None
                value = self.fs

            self._fsPlot = value

            if self._init: self._refresh()

    def __init__(self, fs, hdf5Node=None, plotWidget=None, lineCount=1,
            yScale=1, yOffset=0, yGap=1, color=list('rgbcmyk'), timeBase=False,
            chunkSize=1, flFilter=0, fhFilter=0, fsPlot=5e3,
            label=None, labelOffset=0):
        '''
        Args:
            fs (float): Actual sampling frequency of the channel in Hz.
            hdf5Node (str): Path of the node to store data in HDF5 file.
                Defaults to None.
            plotWidget (ChannelPlotWidget): Defaults to None.
            lineCount (int): Number of lines. Defaults to 1.
            timeBase (bool): Whether to set current channel as timeBase of the
                plotWidget. Defaults to False.
            yScale (float): In-place scaling factor for indivudal lines.
                Defaults to 1.
            yOffset (float): Defaults to 0.
            yGap (float): Defaults to 10.
            color (list of string or tuple): Defaults to list('rgbcmyk').
            chunkSize (float): In seconds. Defaults to 1.
            flFilter (float): Lower cutoff frequency.
            fhFilter (float): Higher cutoff frequency.
            fsPlot (float): In Hz. Defaults to 5e3.
            label (str): ...
            labelOffset (float): ...
        '''

        self._init        = False
        self._refreshLock = threading.Lock()

        self._fs          = fs
        self._hdf5Node    = hdf5Node
        self._plotWidget  = plotWidget
        self._lineCount   = lineCount
        self._yScale      = yScale
        self._yOffset     = yOffset
        self._yGap        = yGap
        self._color       = color if isinstance(color, list) else [color]
        self._chunkSize   = chunkSize
        self.flFilter     = flFilter
        self.fhFilter     = fhFilter
        self.fsPlot       = fsPlot

        buffer1Size       = int(self.plotWidget.xRange*self.fs    *1.2)
        buffer2Size       = int(self.plotWidget.xRange*self.fsPlot*1.2)
        self._buffer1     = misc.CircularBuffer((lineCount, buffer1Size))
        self._buffer2     = misc.CircularBuffer((lineCount, buffer2Size))

        self._init        = True

        self._refresh()

        self._processThread = threading.Thread(target=self._processLoop)
        self._processThread.daemon = True
        self._processThread.start()

        plotWidget.addChannel(self, label, yOffset+labelOffset, timeBase)

        if hdf5Node is not None:
            if hdf5.contains(hdf5Node):
                raise NameError('HDF5 node %s already exists' % hdf5Node)
            hdf5.createEArray(hdf5Node, tb.Float64Atom(), (0,lineCount),
                '', filters, expectedrows=fs*60*30)    # 30 minutes
            hdf5.setNodeAttr(hdf5Node, 'fs', fs)

        self.initPlot()

    def _processLoop(self):
        try:
            while True:
                self._buffer1.wait()

                with self._refreshLock:
                    # read input buffer
                    with self._buffer1:
                        ns = self._buffer1.nsWritten
                        if self._dsFactor:
                            ns = ns//self._dsFactor*self._dsFactor
                        if ns <= self._buffer1.nsRead: continue
                        data = self._buffer1.read(to=ns).copy()

                    # downsample
                    if self._dsFactor:
                        data = data.reshape(
                            (self.lineCount, -1, self._dsFactor))
                        data = data.mean(axis=2)

                    # IIR filter
                    if self._baFilter:
                        data, self._ziFilter = sp.signal.lfilter(
                            *self._baFilter, data, zi=self._ziFilter)

                    # write to output buffer
                    with self._buffer2:
                        self._buffer2.write(data)

        except:
            log.exception('Failed to process data, stopping process thread')

    def _refreshFilter(self):
        # b = sp.signal.firwin(201, fband, fs=self._fsPlot, pass_zero=False)
        # a = [1]
        if self.flFilter and self.fhFilter:
            self._baFilter = sp.signal.butter(6, np.array([self.flFilter,
                self.fhFilter])/self.fsPlot, 'bandpass')
        elif self.flFilter:
            self._baFilter = sp.signal.butter(6, self.flFilter/self.fsPlot,
                'highpass')
        elif self.flFilter:
            self._baFilter = sp.signal.butter(6, self.fhFilter/self.fsPlot,
                'lowpass')
        else:
            self._baFilter = None

        if self._baFilter:
            zi = sp.signal.lfilter_zi(*self._baFilter)
            self._ziFilter = np.zeros((self.lineCount, len(zi)))

    def _refresh(self):
        nsRange = int(self.plotWidget.xRange*self.fs)
        self._buffer1.nsRead  = self._buffer1.nsRead  // nsRange * nsRange
        nsRange = int(self.plotWidget.xRange*self.fsPlot)
        self._buffer2.nsWritten = self._buffer2.nsWritten // nsRange * nsRange
        # self._chunkLast = 0

    def initPlot(self):
        # prepare plotting chunks
        chunkCount = int(np.ceil(self.plotWidget.xRange/self._chunkSize))
        self._curves = [[None]*chunkCount for i in range(self.lineCount)]
        for i in range(self.lineCount):
            # line = pg.InfiniteLine(i*self._yGap,0, pen=pg.mkPen((150,150,150),
            #     style=QtCore.Qt.DashLine, cosmetic=True))
            # self.plotWidget.addItem(line)
            for j in range(chunkCount):
                self._curves[i][j] = self.plotWidget.plot([], [],
                    pen=self._color[i % len(self._color)])

    def append(self, data):
        '''
        Args:
            data (numpy.array): 1D for single line or 2D for multiple lines
                with the following format: lines x samples
        '''
        # some format checking
        if data.ndim == 1: data = np.array([data])
        if data.ndim != 2: raise ValueError('Need 1 or 2-dimensional data')
        if data.shape[0] != self.lineCount: raise ValueError('Size of first '
            'dimension of data should match line count')

        # dump new data to HDF5 file
        if self._hdf5Node is not None:
            hdf5.appendArray(self._hdf5Node, data.transpose())

        # keep a local copy of data in a circular buffer
        # for online processing and fast plotting
        with self._buffer1:
            self._buffer1.write(data)

    def updatePlot(self, ts=None, tsMin=None, nextWindow=False):
        if not self._buffer2.updated: return

        with self._refreshLock, self._buffer2:
            nsRange      = int(self.plotWidget.xRange*self._fsPlot)
            chunkSamples = int(self._chunkSize*self._fsPlot)

            nsFrom       = self._buffer2.nsRead
            nsFromMin    = int(nsFrom//nsRange*nsRange)
            chunkFrom    = int((nsFrom-nsFromMin)//chunkSamples)

            nsTo         = self._buffer2.nsWritten
            nsToMin      = int(nsTo//nsRange*nsRange)
            chunkTo      = int((nsTo-nsToMin)//chunkSamples)

            # when plot advances to next time window
            # if nextWindow: chunkFrom = chunkTo = 0
            if chunkFrom > chunkTo: chunkFrom = 0

            for chunk in range(chunkFrom, chunkTo+1):
                nsChunkFrom  = nsToMin + chunk*chunkSamples
                nsChunkTo    = min(nsTo, nsChunkFrom+chunkSamples)
                if nsChunkFrom == nsChunkTo: continue
                time         = np.arange(nsChunkFrom, nsChunkTo) / self._fsPlot
                data         = self._buffer2.read(nsChunkFrom, nsChunkTo)
                for line in range(self.lineCount):
                    self._curves[line][chunk].setData(time,
                        data[line,:]*self._yScale
                        + self._yOffset + line*self._yGap)

    def refresh(self):
        with self._refreshLock:
            self._refresh()


# class AnalogGenerator():
#     _gps = 40    # generate per second
#
#     def __init__(self, *args):
#         self._channels   = args
#         self._nsLast     = 0
#         self._threadStop = False
#         self._thread     = None
#
#     def _loop(self):
#         try:
#             while not self._threadStop:
#                 ts = (dt.datetime.now()-self._timeStart).total_seconds()
#                 for channel in self._channels:
#                     nsTotal = int(ts*channel.fs)+1
#                     ns = nsTotal-channel.ns
#                     data = np.empty((ns,channel.channels))
#                     for i in range(channel.channels):
#                         data[i,:] = np.arange(channel.ns,nsTotal)/channel.fs
#                     data = np.sin(2*np.pi*5*data)+np.random.randn(*data.shape)
#                     # data = np.random.randn(*data.shape)
#                     channel.append(data)
#
#                 time.sleep(1/self._gps)
#         except:
#             log.exception('Failed to generate buffer')
#
#     def start(self):
#         if self._thread is not None:
#             raise NotImplementedError('Generator restart not implemented')
#         self._timeStart = dt.datetime.now()
#         self._thread = threading.Thread(target=self._loop)
#         self._thread.daemon = True
#         self._thread.start()
#
#     def stop(self):
#         if self._thread is not None:
#             self._threadStop = True
#             log.info('Waiting for generator thread to join')
#             self._thread.join()


class BaseEpochChannel(BaseChannel):
    '''Base channel for storage of epochs in the HDF5 file format.

    An epoch is a pair of timesamples indicating the start and stop times of a
    specific event in seconds. This class is only responsible for storage of
    epochs in the HDF5 file format and has no graphical represenation. Plotting
    funcionality can be added by subclasses.
    '''
    def __init__(self, hdf5Node=None):
        self._hdf5Node   = hdf5Node
        self._data       = []
        self._dataAdded  = []
        self._lock       = threading.Lock()

        if hdf5Node is not None:
            if hdf5.contains(hdf5Node):
                raise NameError('HDF5 node %s already exists' % hdf5Node)
            hdf5.createEArray(hdf5Node, tb.Float64Atom(), (0,2), '',
                filters, expectedrows=200)

    def start(self, ts):
        # only start epoch
        if len(self._data) and self._data[-1][1] is None:
            raise ValueError('Last epoch has not been stopped')

        # cache a partial epoch
        with self._lock:
            self._data += [[ts, None]]
            self._dataAdded = True

    def stop(self, ts):
        # only stop epoch (if any)
        if not len(self._data) or self._data[-1][1] is not None:
            # raise ValueError('No previously started epoch to stop')
            return

        # dump epoch to file
        if self._hdf5Node is not None:
            hdf5.appendArray(self._hdf5Node,
                np.array([[self._data[-1][0],ts]]))

        # complete the last partial epoch in cached data
        with self._lock:
            self._data[-1][1] = ts
            self._dataAdded = True

    def append(self, start, stop):
        # full epoch
        if len(self._data) and self._data[-1][1] is None:
            raise ValueError('Last epoch has not been stopped')

        # dump epoch to file
        if self._hdf5Node is not None:
            hdf5.appendArray(self._hdf5Node, np.array([[start,stop]]))

        # keep a cached copy of epochs for plotting
        with self._lock:
            self._data += [[start, stop]]
            self._dataAdded = True


class SymbEpochChannel(BaseEpochChannel):
    '''Epoch channel represented graphically with two symbols.'''
    def __init__(self, hdf5Node=None, plotWidget=None, yOffset=0,
            color=(0,255,0), symbolStart='t1', symbolStop='t', symbolSize=15,
            label=None):
        super().__init__(hdf5Node)

        self._plotWidget   = plotWidget
        plotWidget.addChannel(self, label, yOffset)
        self._yOffset      = yOffset

        # prepare scatter plots, one for epoch start and one for epoch stop
        kwargs = dict(pen=None, symbolPen=None, symbolBrush=pg.mkBrush(color),
            symbolSize=symbolSize)
        self._scatterStart = plotWidget.plot([], symbol=symbolStart, **kwargs)
        self._scatterStop  = plotWidget.plot([], symbol=symbolStop , **kwargs)

    def updatePlot(self, ts=None, tsMin=None, nextWindow=False):
        with self._lock:
            if not self._dataAdded: return

            starts = []
            stops  = []
            remEpochs = []    # list of epochs to remove from cached data

            for i in range(len(self._data)):
                # get epoch bounds
                start = self._data[i][0]
                stop  = self._data[i][1]

                if stop is not None and stop < tsMin:
                    remEpochs += [self._data[i]]
                else:
                    starts += [start]
                    if stop is not None:
                        stops += [stop]

            self._scatterStart.setData(starts, [self._yOffset]*len(starts))
            self._scatterStop .setData(stops , [self._yOffset]*len(stops ))

            # remove old epochs from cached data
            for epoch in remEpochs:
                self._data.remove(epoch)

            self._dataAdded = False


class RectEpochChannel(BaseEpochChannel):
    '''Epoch channel represented graphically by rectangle.'''

    def __init__(self, hdf5Node=None, plotWidget=None, yOffset=0, yRange=1,
            color=(0,0,255,100), label=None):
        super().__init__(hdf5Node)

        self._plotWidget = plotWidget
        plotWidget.addChannel(self, label, yOffset+yRange/2)
        self._yOffset = yOffset
        self._yRange  = yRange
        self._color   = color

        self._rects   = []

    def updatePlot(self, ts=None, tsMin=None, nextWindow=False):
        with self._lock:
            # the statement below has a bug since it doesn't check whether a
            # rect has been already added for the last epoch or not
            # partialEpoch = len(self._data)>0 and (self._data[-1][1] is None or
            #     ts<self._data[-1][1] or self._rects[-1].rect().right()
            #         <self._data[-1][1])
            #
            # if not self._dataAdded or not partialEpoch: return

            remEpochs = []    # list of epochs to remove from cached data
            remRects  = []    # list of rects to remove from rect list and plot

            for i in range(len(self._data)):
                # get epoch bounds
                start = self._data[i][0]
                stop  = self._data[i][1]
                if ts < start:
                    continue
                if stop is None or ts<stop:
                    stop = ts

                # add a rect for new epoch
                if len(self._rects)<=i:
                    rect = QtGui.QGraphicsRectItem(start, self._yOffset,
                        stop-start, self._yRange)
                    rect.setPen(QtGui.QPen(QtCore.Qt.NoPen))
                    rect.setBrush(pg.mkBrush(*self._color))
                    self._plotWidget.addItem(rect)
                    self._rects += [rect]

                # resize current epoch
                elif self._rects[i].rect().right()<stop:
                    self._rects[i].setRect(start, self._yOffset,
                        stop-start, self._yRange)

                # prepare old epoch for removal
                elif stop<tsMin:
                    remEpochs += [self._data[i]]
                    remRects  += [self._rects[i]]

            # remove old epochs from cached data
            for (epoch, rect) in zip(remEpochs, remRects):
                self._data.remove(epoch)
                self._rects.remove(rect)
                self._plotWidget.removeItem(rect)
