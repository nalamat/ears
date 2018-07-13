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
# complib: zlib, lzo, bzip2, blosc, blosc:blosclz, blosc:lz4,
# blosc:lz4hc, blosc:snappy, blosc:zlib, blosc:zstd
# complevel: 0 (no compression) to 9 (maximum compression)
hdf5Filters = tb.Filters(complib='zlib', complevel=1)


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
    '''A Qt widget for managing and plotting multiple channels online.'''

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
    # @property
    # def ts(self):
    #     if self.timeBase is None:
    #         raise AttributeError('No timeBase has been set')
    #     return self.timeBase.ts

    # min time in the current plot window
    # @property
    # def tsMin(self):
    #     return int(self.ts//self.tsRange*self.tsRange)

    @property
    def timeBase(self):
        return self._timeBase

    @timeBase.setter
    def timeBase(self, value):
        if not isinstance(value, AnalogChannel):
            raise TypeError('`timeBase` can only be an instance '
                'of `AnalogChannel`')
        self._timeBase = value

    def __init__(self,
            xLimits=(0,10), xGrid=1, xLabel=None, xRange=10, xTicksFormat=None,
            yLimits=(-1,1), yGrid=1, yLabel=None, timePlot=True, fps=60,
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
            fps (float): ...

        '''

        super().__init__(*args, **kwargs, axisItems={'left':Axis('left')})

        # in a time plot, xRange ovverides xLimits
        if timePlot:
            # pass
            xLimits = (0, xRange)
            if xLabel is None:
                xLabel = 'Time (min:sec)'
            if xTicksFormat is None:
                xTicksFormat = lambda x: '%02d:%02d' % (x//60,x%60)
        # in a normal plot, xLimits override xRange
        else:
            # pass
            xRange = xLimits[1]-xLimits[0]
            if xTicksFormat is None:
                xTicksFormat = lambda x: str(x)

        # should not change after __init__
        self._yLimits       = yLimits
        self._yGrid         = yGrid
        self._xLimits       = xLimits
        self._xGrid         = xGrid
        self._xRange        = xRange
        self._xTicksFormat  = xTicksFormat
        self._timePlot      = timePlot
        self._fps           = fps
        self._calculatedFPS = 0

        self._channels      = []
        self._updateTimes   = []    # hold last update times for FPS calculation
        self._timeBase      = None
        self._tsMinLast     = None

        self._timer         = QtCore.QTimer()
        self._timer.timeout.connect(self.updatePlot)
        self._timer.setInterval(1000/self.fps)

        self.setBackground('w')
        self.setMouseEnabled(False, False)
        self.setMenuEnabled(False)
        self.setClipToView(False)
        self.disableAutoRange()
        self.setRange(xRange=xLimits, yRange=yLimits, padding=0)
        self.setLimits(xMin=xLimits[0], xMax=xLimits[1],
            yMin=yLimits[0], yMax=yLimits[1])
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
        if xLabel is not None:
            self.getAxis('top' ).setLabel('<br />' + xLabel)
        if yLabel is not None:
            self.getAxis('left').setLabel('<br />' + yLabel)

        gridMajorPen = pg.mkPen((150,150,150), style=QtCore.Qt.DashLine,
            cosmetic=True)
        gridMinorPen = pg.mkPen((150,150,150), style=QtCore.Qt.DotLine,
            cosmetic=True)

        # init x grid (vertical)
        self._xGridMajor = [None] * int(np.ceil(xRange/xGrid)-1)
        self._xGridMinor = [None] * int(np.ceil(xRange/xGrid))
        for i in range(len(self._xGridMajor)):
            self._xGridMajor[i] = pg.InfiniteLine(None, 90, pen=gridMajorPen)
            self.addItem(self._xGridMajor[i])
        for i in range(len(self._xGridMinor)):
            self._xGridMinor[i] = pg.InfiniteLine(None, 90, pen=gridMinorPen)
            self.addItem(self._xGridMinor[i])

        # draw x grid and ticks
        self.getAxis('top').setStyle(tickLength=0)
        self.updateXAxis()

        # draw y grid (horizontal)
        if yGrid is None: yGridAt = []
        elif misc.listLike(yGrid): yGridAt = yGrid
        else: yGridAt = range(yLimits[0]+yGrid, yLimits[1], yGrid)
        for y in yGridAt:
            line = pg.InfiniteLine(y, 0, pen=gridMajorPen)
            self.addItem(line)

        # draw y ticks
        yMajorTicks = [(y, '') for y in yGridAt]
        yMinorTicks = []
        self.getAxis('left').setTicks([yMajorTicks, yMinorTicks])
        self.getAxis('left').setStyle(tickLength=0)

    def addChannel(self, channel, label=None, labelOffset=0):
        if not isinstance(channel, BaseChannel):
            raise TypeError('Channel should be an instance of BaseChannel'
                'or one of its subclasses')
        if channel in self._channels:
            raise RuntimeError('Cannot add a channel twice')

        self._channels.append(channel)
        if label:
            if not misc.listLike(label, False): label       = [label      ]
            if not misc.listLike(labelOffset ): labelOffset = [labelOffset]
            if len(label) != len(labelOffset):
                raise ValueError('Number of `label`s (%d) should match number '
                    'of `labelOffset`s (%d)' % (len(label), len(labelOffset)))
            ticks = self.getAxis('left')._tickLevels
            for i in range(len(label)):
                ticks = [ticks[0] + [(labelOffset[i],' '+label[i])], ticks[1]]
            self.getAxis('left').setTicks(ticks)
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
            # log.info('Updating plot')

            # calculate actual fps
            updateTime = (dt.datetime.now()-self._updateLast).total_seconds()
            self._updateTimes.append(updateTime)
            if len(self._updateTimes)>60: self._updateTimes.pop(0)
            updateMean = np.mean(self._updateTimes)
            fps = 1/updateMean if updateMean else 0
            self._calculatedFPS = fps
            self._updateLast = dt.datetime.now()

            if self._timePlot:
                ts         = None
                tsMin      = None
                nextWindow = False

                if self._timeBase is not None:
                    ts = self.timeBase.ts
                    tsMin = int(ts//self.xRange*self.xRange)

                    # determine when plot advances to next time window
                    if tsMin != self._tsMinLast:
                        self.setLimits(xMin=tsMin, xMax=tsMin+self.xRange)

                        self.updateXAxis(tsMin)

                        log.debug('Advancing to next plot window at %d', tsMin)
                        self._tsMinLast = tsMin
                        nextWindow = True

                # update all channels
                for channel in self._channels:
                    channel.updatePlot(ts, tsMin, nextWindow)
            else:
                # update all channels
                for channel in self._channels:
                    channel.updatePlot()

        except:
            log.exception('Failed to update plot, stopping plot timer')
            self._timer.stop()

    def updateXAxis(self, xMin=None):
        '''Update X ticks and grid'''
        if xMin is None:
            xMin = self._xLimits[0]

        # update vertical grid
        for i in range(len(self._xGridMajor)):
            self._xGridMajor[i].setValue(xMin+(i+1)*self._xGrid)
        for i in range(len(self._xGridMinor)):
            self._xGridMinor[i].setValue(xMin+(i+.5)*self._xGrid)

        xMajorTicks = [(t, self._xTicksFormat(t))
            for t in np.arange(xMin,xMin+self.xRange+self._xGrid/2,self._xGrid)]
        xMinorTicks = [(t, self._xTicksFormat(t))
            for t in np.arange(xMin+self._xGrid/2,xMin+self.xRange,self._xGrid)]
        self.getAxis('top').setTicks([xMajorTicks, xMinorTicks])


class BaseChannel():
    '''Base class for all channels.

    Contains not much functional code, mostly for class hierarchy purposes.
    '''

    @property
    def plotWidget(self):
        return self._plotWidget

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, value):
        if value and not isinstance(value, BaseChannel):
            raise TypeError('`source` should be of type `BaseChannel`')
        self._source = value
        if value:
            value._sink  = self

    @property
    def sink(self):
        return self._sink

    @sink.setter
    def sink(self, value):
        if value and not isinstance(value, BaseChannel):
            raise TypeError('`sink` should be of type `BaseChannel`')
        self._sink    = value
        if value:
            value._source = self

    def __init__(self, plotWidget=None, label=None, labelOffset=0,
            source=None):
        self._plotWidget  = plotWidget
        self._label       = label
        self._labelOffset = labelOffset
        self.source       = source
        self._sink        = None

        if plotWidget:
            plotWidget.addChannel(self, label, labelOffset)


class AnalogChannel(BaseChannel):
    '''All-in-one class for storage and plotting of multiline analog signals.

    Utilizes the `hdf5` module for storage in HDF5 file format.

    Needs to be paired with a ChannelPlotWidget for plotting.
    '''

    @property
    def refreshBlock(self):
        return self._refreshBlock

    @refreshBlock.setter
    def refreshBlock(self, value):
        self._refreshBlock = value

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
    def yScale(self):
        return self._yScale

    @yScale.setter
    def yScale(self, value):
        # verify
        if hasattr(self, '_yScale') and self._yScale == value:
            return

        with self._refreshLock:
            # save
            self._yScale = value

            # apply
            if not self._refreshBlock:
                self._refresh()

    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, value):
        # verify
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError('`filter` should be a tuple of 2')
        if hasattr(self, '_filter') and self._filter == value:
            return

        with self._refreshLock:
            # save
            self._filter = value

            # prep filter
            fs = self.fs
            fl, fh = value
            if (fl and 0<fl and fh and fh<fs):
                filterBA = sp.signal.butter(6, [fl/fs,fh/fs], 'bandpass')
            elif fl and 0<fl:
                filterBA = sp.signal.butter(6, fl/fs, 'highpass')
            elif fh and fh<fs:
                filterBA = sp.signal.butter(6, fh/fs, 'lowpass')
            else:
                filterBA = None

            self._filterBA = filterBA
            if self._filterBA:
                zi = sp.signal.lfilter_zi(*self._filterBA)
                self._filterZI = np.zeros((self.lineCount, len(zi)))

            # apply
            if not self._refreshBlock:
                self._refresh()

    @property
    def fsPlot(self):
        return self._fsPlot

    @property
    def linesVisible(self):
        return self._linesVisible

    @linesVisible.setter
    def linesVisible(self, value):
        # verify
        if not isinstance(value, tuple):
            raise TypeError('Value should be a tuple')
        if len(value) != self.lineCount:
            raise ValueError('Value should have exactly `lineCount` elements')
        for v in value:
            if not isinstance(v, bool):
                raise ValueError('All elements must be `bool` instances')

        if hasattr(self, '_linesVisible') and self._linesVisible == value:
            return

        with self._refreshLock:
            # save
            self._linesVisible = value

            # apply
            if hasattr(self, '_curves'):
                for i in range(self.lineCount):
                    for curve in self._curves[i]:
                        curve.setVisible(value[i])

    @property
    def linesGrandMean(self):
        return self._linesGrandMean

    @linesGrandMean.setter
    def linesGrandMean(self, value):
        # verify
        if not isinstance(value, tuple):
            raise TypeError('Value should be a tuple')
        if len(value) != self.lineCount:
            raise ValueError('Value should have exactly `lineCount` elements')
        for v in value:
            if not isinstance(v, bool):
                raise ValueError('All elements must be `bool` instances')
        if hasattr(self, '_linesGrandMean') and self._linesGrandMean == value:
            return

        with self._refreshLock:
            # save
            self._linesGrandMean = value

            # apply
            if not self._refreshBlock:
                self._refresh()

    def __init__(self, fs, plotWidget=None, label=None, labelOffset=0,
            source=None, hdf5Node=None, lineCount=1, yScale=1, yOffset=0,
            yGap=1, color=list('rgbcmyk'), chunkSize=.5, filter=(None,None),
            fsPlot=5e3, grandMean=False):
        '''
        Args:
            fs (float): Actual sampling frequency of the channel in Hz.
            plotWidget (ChannelPlotWidget): Defaults to None.
            label (str or list of str): ...
            labelOffset (float): ...
            source (BaseChannel): ...
            hdf5Node (str): Path of the node to store data in HDF5 file.
                Defaults to None.
            lineCount (int): Number of lines. Defaults to 1.
            yScale (float): In-place scaling factor for indivudal lines.
                Defaults to 1.
            yOffset (float): Defaults to 0.
            yGap (float): The gap between multiple lines. Defaults to 10.
            color (list of str or tuple): Defaults to list('rgbcmyk').
            chunkSize (float): In seconds. Defaults to 0.5.
            filter (tuple of 2 floats): Lower and higher cutoff frequencies.
            fsPlot (float): Plotting sampling frequency in Hz. Defaults to 5e3.
        '''

        labelOffset = np.arange(lineCount)*yGap+yOffset+labelOffset
        super().__init__(plotWidget, label, labelOffset, source)

        self._refreshBlock   = True
        self._refreshLock    = threading.Lock()

        self._fs             = fs
        self._hdf5Node       = hdf5Node
        self._lineCount      = lineCount
        self.yScale          = yScale
        self._yOffset        = yOffset
        self._yGap           = yGap
        self._color          = color if isinstance(color, list) else [color]
        self._chunkSize      = chunkSize
        self.filter          = filter
        self.linesVisible    = (True,) * lineCount
        self.linesGrandMean  = (grandMean,) * lineCount

        # calculate an integer downsampling factor base on `fsPlot`
        if fsPlot and fsPlot != self.fs:
            # downsampling factor, i.e. number of consecutive samples to
            # average, must be integer
            self._dsFactor   = round(self.fs/fsPlot)
            # recalculate the plotting sampling frequency
            self._fsPlot     = self.fs / self._dsFactor
        else:
            self._dsFactor   = None
            self._fsPlot     = self.fs

        buffer1Size          = int(self.plotWidget.xRange*self.fs    *1.2)
        buffer2Size          = int(self.plotWidget.xRange*self.fsPlot*1.2)
        # self._bufferX        = misc.CircularBuffer((lineCount, buffer1Size))
        self._buffer1        = misc.CircularBuffer((lineCount, buffer1Size))
        self._buffer2        = misc.CircularBuffer((lineCount, buffer2Size))

        self._refreshBlock   = False
        self._refresh()

        self._processThread = threading.Thread(target=self._processLoop)
        self._processThread.daemon = True
        self._processThread.start()

        if hdf5Node is not None:
            if hdf5.contains(hdf5Node):
                raise NameError('HDF5 node %s already exists' % hdf5Node)
            hdf5.createEArray(hdf5Node, tb.Float32Atom(), (0,lineCount),
                '', hdf5Filters, expectedrows=fs*60*30)    # 30 minutes
            hdf5.setNodeAttr(hdf5Node, 'fs', fs)

        self.initPlot()

    def _processLoop(self):
        try:
            while True:
                self._buffer1.wait()

                with self._refreshLock:
                    # read input buffer
                    with self._buffer1:
                        ns = self._buffer1.nsWritten-self._buffer1.nsRead
                        if self._dsFactor:
                            ns = ns//self._dsFactor*self._dsFactor
                        if ns <= 0: continue
                        ns += self._buffer1.nsRead
                        data = self._buffer1.read(to=ns).copy()

                    dataCopy = data.copy()
                    for line in range(data.shape[0]):
                        linesMask = np.array(self._linesGrandMean)
                        if not linesMask[line]: continue
                        linesMask[line] = False
                        if not linesMask.any(): continue
                        data[line,:] -= dataCopy[linesMask,:].mean(axis=0)

                    # IIR filter
                    if self._filterBA:
                        data, self._filterZI = sp.signal.lfilter(
                            *self._filterBA, data, zi=self._filterZI)

                    # downsample
                    if self._dsFactor:
                        data = data.reshape(
                            (self.lineCount, -1, self._dsFactor))
                        data = data.mean(axis=2)

                    # write to output buffer
                    with self._buffer2:
                        self._buffer2.write(data)

        except:
            log.exception('Failed to process data, stopping process thread')

    def _refresh(self):
        # reset buffers to the beginning of the current plot window
        nsRange = int(self.plotWidget.xRange*self.fs)
        self._buffer1.nsRead    = self._buffer1.nsRead    // nsRange * nsRange
        nsRange = int(self.plotWidget.xRange*self.fsPlot)
        self._buffer2.nsWritten = self._buffer2.nsWritten // nsRange * nsRange
        # self._chunkLast = 0

    def initPlot(self):
        # prepare plotting chunks
        self._chunkCount = int(np.ceil(self.plotWidget.xRange/self._chunkSize))
        self._curves = [[None]*self._chunkCount for i in range(self.lineCount)]
        for i in range(self.lineCount):
            for j in range(self._chunkCount):
                self._curves[i][j] = self.plotWidget.plot([], [],
                    pen=self._color[i % len(self._color)])
                self._curves[i][j].setVisible(self.linesVisible[i])

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
            # if self._bufferX.nsAvailable+data.shape[-1]>self._bufferX.shape[-1]:
            #     oldData = self._bufferX.read()
            #     hdf5.appendArray(self._hdf5Node, oldData.transpose())
            # self._bufferX.write(data)

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
            nsFromMin    = nsFrom // nsRange * nsRange
            chunkFrom    = (nsFrom - nsFromMin) // chunkSamples

            nsTo         = self._buffer2.nsWritten
            nsToMin      = nsTo // nsRange * nsRange
            chunkTo      = (nsTo - nsToMin) // chunkSamples
            chunkTo      = min(chunkTo, self._chunkCount-1)

            # when plot advances to next time window
            # if nextWindow: chunkFrom = chunkTo = 0
            if chunkTo < chunkFrom: chunkFrom = 0

            try:
                for chunk in range(chunkFrom, chunkTo+1):
                    nsChunkFrom  = nsToMin + chunk*chunkSamples
                    nsChunkTo    = min(nsTo, nsChunkFrom+chunkSamples)
                    if nsChunkFrom == nsChunkTo: continue
                    time         = np.arange(nsChunkFrom,
                                   nsChunkTo) / self._fsPlot
                    data         = self._buffer2.read(nsChunkFrom, nsChunkTo)
                    for line in range(self.lineCount):
                        self._curves[line][chunk].setData(time,
                            data[line,:]*self._yScale
                            + self._yOffset + line*self._yGap)
            except:
                log.info('nsRange %d, chunkSamples %d, nsFrom %d, '
                    'nsFromMin %d, chunkFrom %d, nsTo %d, nsToMin %d, '
                    'chunkTo %d, chunk %d, line %d', nsRange, chunkSamples,
                    nsFrom, nsFromMin, chunkFrom, nsTo, nsToMin, chunkTo,
                    chunk, line)
                raise

    def refresh(self):
        with self._refreshLock:
            self._refresh()


class FFTChannel(BaseChannel):

    def __init__(self, fs, plotWidget=None, label=None, labelOffset=0,
        source=None, hdf5Node=None, lineCount=1, yScale=1, yOffset=0,
        yGap=1, color=list('rgbcmyk'), chunkSize=.5, filter=(None,None),
        fsPlot=5e3):
        '''
        Args:
        fs (float): Actual sampling frequency of the channel in Hz.
        plotWidget (ChannelPlotWidget): Defaults to None.
        label (str or list of str): ...
        labelOffset (float): ...
        source (BaseChannel): ...
        hdf5Node (str): Path of the node to store data in HDF5 file.
        Defaults to None.
        lineCount (int): Number of lines. Defaults to 1.
        yScale (float): In-place scaling factor for indivudal lines.
        Defaults to 1.
        yOffset (float): Defaults to 0.
        yGap (float): The gap between multiple lines. Defaults to 10.
        color (list of str or tuple): Defaults to list('rgbcmyk').
        chunkSize (float): In seconds. Defaults to 0.5.
        filter (tuple of 2 floats): Lower and higher cutoff frequencies.
        fsPlot (float): Plotting sampling frequency in Hz. Defaults to 5e3.
        '''

        labelOffset = np.arange(lineCount)*yGap+yOffset+labelOffset
        super().__init__(plotWidget, label, labelOffset, source)

        self._refreshBlock = True
        self._refreshLock  = threading.Lock()

        self._fs           = fs
        self._hdf5Node     = hdf5Node
        self._lineCount    = lineCount
        self.yScale        = yScale
        self._yOffset      = yOffset
        self._yGap         = yGap
        self._color        = color if isinstance(color, list) else [color]
        self._chunkSize    = chunkSize
        self.filter        = filter
        self.fsPlot        = fsPlot

        buffer1Size        = int(self.plotWidget.xRange*self.fs    *1.2)
        buffer2Size        = int(self.plotWidget.xRange*self.fsPlot*1.2)
        # self._bufferX      = misc.CircularBuffer((lineCount, buffer1Size))
        self._buffer1      = misc.CircularBuffer((lineCount, buffer1Size))
        self._buffer2      = misc.CircularBuffer((lineCount, buffer2Size))

        self._refreshBlock = False

        self._refresh()

        self._processThread = threading.Thread(target=self._processLoop)
        self._processThread.daemon = True
        self._processThread.start()

        if hdf5Node is not None:
            if hdf5.contains(hdf5Node):
                raise NameError('HDF5 node %s already exists' % hdf5Node)
            hdf5.createEArray(hdf5Node, tb.Float64Atom(), (0,lineCount),
                '', hdf5Filters, expectedrows=fs*60*30)    # 30 minutes
            hdf5.setNodeAttr(hdf5Node, 'fs', fs)

        self.initPlot()



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

    def __init__(self, plotWidget=None, label=None, labelOffset=0, source=None,
            hdf5Node=None):

        super().__init__(plotWidget, label, labelOffset, source)

        self._hdf5Node   = hdf5Node
        self._data       = []
        self._dataAdded  = []
        self._lock       = threading.Lock()

        if hdf5Node is not None:
            if hdf5.contains(hdf5Node):
                raise NameError('HDF5 node %s already exists' % hdf5Node)
            hdf5.createEArray(hdf5Node, tb.Float64Atom(), (0,2), '',
                hdf5Filters, expectedrows=200)

    def start(self, ts):
        # only start epoch
        if len(self._data) and self._data[-1][1] is None:
            raise ValueError('Last epoch has not been stopped')

        # cache a partial epoch
        with self._lock:
            self._data += [[ts, None]]
            self._dataAdded = True

        if self._sink:
            self._sink.start(ts)

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

        if self._sink:
            self._sink.stop(ts)

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

        if self._sink:
            self._sink.append(start, stop)


class SymbEpochChannel(BaseEpochChannel):
    '''Epoch channel represented graphically with two symbols.'''

    def __init__(self, plotWidget=None, label=None, labelOffset=0, source=None,
            hdf5Node=None, yOffset=0, color=(0,255,0), symbolStart='t1',
            symbolStop='t', symbolSize=15):

        labelOffset += yOffset
        super().__init__(plotWidget, label, labelOffset, source, hdf5Node)

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

    def __init__(self, plotWidget=None, label=None, labelOffset=0, source=None,
            hdf5Node=None, yOffset=0, yRange=1, color=(0,0,255,100)):
        labelOffset += yOffset + yRange/2
        super().__init__(plotWidget, label, labelOffset, source, hdf5Node)

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
