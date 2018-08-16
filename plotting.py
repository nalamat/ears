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

import hdf5
import misc
import pipeline


log = logging.getLogger(__name__)
# complib: zlib, lzo, bzip2, blosc, blosc:blosclz, blosc:lz4,
# blosc:lz4hc, blosc:snappy, blosc:zlib, blosc:zstd
# complevel: 0 (no compression) to 9 (maximum compression)
hdf5Filters = tb.Filters(complib='zlib', complevel=1)


class ScrollingPlotWidget(pg.PlotWidget):
    '''A widget for plotting analog traces and epochs in a scrolling window.'''

    @property
    def fps(self):
        '''Target rate of plotting in frames per second.'''
        return self._fps

    @property
    def measuredFPS(self):
        '''Actual rate of plotting in frames per second.'''
        return self._measuredFPS

    @property
    def xRange(self):
        '''The x-axis (time) range of the plot window in seconds.'''
        return self._xRange

    @property
    def timeBase(self):
        '''An instance of AnalogChannel to use as time base for determining
        when to advance to the next plot window.'''
        return self._timeBase

    @timeBase.setter
    def timeBase(self, value):
        if not isinstance(value, (AnalogChannel, AnalogPlot)):
            raise TypeError('`timeBase` can only be an instance '
                'of `AnalogChannel`')
        self._timeBase = value

    def __init__(self, xRange=10, xGrid=1, yLimits=(-1,1), yGrid=1,
            yLabel=None, fps=60, *args, **kwargs):
        '''
        Args:
            xRange (float): The x-axis (time) range of the plot window in
                seconds. Defaults to 10.
            xGrid (float): Spacing between vertical grid lines. Defaults to 1.
            yLimits (tuple of 2 floats): Minimum and maximum values shown on the
                y-axis. Defaults to (1,-1).
            yGrid (float or list-like of float): When given a single value,
                determines spacing between the horizontal grid lines. If a list
                of values is given, horizontal grid lines are drawn at y
                intercepts specified by the list. Defaults to 1.
            yLabel (str): Label to show on the y-axis. Defaults to None.
            fps (float): Target plotting rate in frames per second.
                Defaults to 60.
        '''

        super().__init__(*args, **kwargs)

        # should not change after __init__
        self._xRange        = xRange
        self._xGrid         = xGrid
        self._yLimits       = yLimits
        self._yGrid         = yGrid
        self._fps           = fps

        self._measuredFPS   = 0
        self._timeBase      = None
        self._updateLast    = None
        self._updateTimes   = []    # keep last update times for measuring FPS
        self._channels      = []
        self._tsMinLast     = None  # timestamp of last updated plot window
        self._xTicksFormat  = lambda x: '%02d:%02d' % (x//60,x%60)

        self._timer         = QtCore.QTimer()
        self._timer.timeout.connect(self.update)
        self._timer.setInterval(1000/self.fps)

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

        # set axis labels
        self.getAxis('top'   ).setLabel('<br />Time (min:sec)')
        if yLabel is not None:
            self.getAxis('left').setLabel('<br />' + yLabel)

        gridMajorPen = pg.mkPen((150,150,150), style=QtCore.Qt.DashLine,
            cosmetic=True)
        gridMinorPen = pg.mkPen((150,150,150), style=QtCore.Qt.DotLine,
            cosmetic=True)

        # init x grid (vertical)
        if xGrid:
            self._xGridMajor = [None] * int(np.ceil(xRange/xGrid)-1)
            self._xGridMinor = [None] * int(np.ceil(xRange/xGrid))
            for i in range(len(self._xGridMajor)):
                line = pg.InfiniteLine(None, 90, pen=gridMajorPen)
                self.addItem(line)
                self._xGridMajor[i] = line
            for i in range(len(self._xGridMinor)):
                line = pg.InfiniteLine(None, 90, pen=gridMinorPen)
                self.addItem(line)
                self._xGridMinor[i] = line

        # draw x grid and ticks
        self.getAxis('top').setStyle(tickLength=0)
        self._updateXAxis()

        # draw y grid (horizontal)
        if yGrid is None: yGridAt = []
        elif misc.listLike(yGrid): yGridAt = yGrid
        else: yGridAt = np.arange(yLimits[0]+yGrid, yLimits[1], yGrid)
        for y in yGridAt:
            line = pg.InfiniteLine(y, 0, pen=gridMajorPen)
            self.addItem(line)

        # draw y ticks
        yMajorTicks = [(y, '') for y in yGridAt]
        yMinorTicks = []
        self.getAxis('left').setTicks([yMajorTicks, yMinorTicks])
        self.getAxis('left').setStyle(tickLength=0)

    def _updateXAxis(self, xMin=0):
        '''Update X ticks and grid'''

        if not self._xGrid: return

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

    def add(self, channel, label=None, labelOffset=0):
        if not isinstance(channel, (AnalogChannel, AnalogPlot, BaseEpochChannel)):
            raise TypeError('Plot should be an instance of AnalogChannel, '
                'BaseEpochChannel or one of their subclasses')
        if channel in self._channels:
            raise RuntimeError('Cannot add the same channel twice')

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
            self.update()
            self._updateLast = None

    def update(self):
        try:
            # log.debug('Updating plot')

            # measure the actual fps
            if self._updateLast is not None:
                updateTime = (dt.datetime.now() -
                    self._updateLast).total_seconds()
                self._updateTimes.append(updateTime)
                if len(self._updateTimes)>60:
                    self._updateTimes.pop(0)
                updateMean = np.mean(self._updateTimes)
                fps = 1/updateMean if updateMean else 0
                self._measuredFPS = fps
            self._updateLast = dt.datetime.now()

            # timing info to send to children
            ts         = None
            tsMin      = None
            nextWindow = False

            if self._timeBase is not None:
                ts = self.timeBase.ts
                tsMin = int(ts//self.xRange*self.xRange)

                # determine when plot advances to next time window
                if tsMin != self._tsMinLast:
                    self.setLimits(xMin=tsMin, xMax=tsMin+self.xRange)

                    self._updateXAxis(tsMin)

                    log.debug('Advancing to next plot window at %d', tsMin)
                    self._tsMinLast = tsMin
                    nextWindow = True

            # update all channels
            for channel in self._channels:
                channel.update(ts, tsMin, nextWindow)

        except:
            log.exception('Failed to update plot, stopping plot timer')
            self._timer.stop()


class FFTPlotWidget(pg.PlotWidget):
    # '''A widget for plotting analog traces and epochs in a scrolling window.'''

    @property
    def fs(self):
        '''Sampling frequency'''
        return self._fs

    @property
    def lineCount(self):
        return self._lineCount

    @property
    def fps(self):
        '''Target rate of plotting in frames per second.'''
        return self._fps

    @property
    def measuredFPS(self):
        '''Actual rate of plotting in frames per second.'''
        return self._measuredFPS

    def __init__(self, fs, lineCount=1, xLimits=(0,5e-3), yLimits=(-5,5),
            yScale=1, yOffset=0, yGap=1, chunk=.2, color=list('rgbcmyk'),
            fps=60, *args, **kwargs):
        '''
        Args:
            xLimits (tuple of 2 floats):
            yLimits (tuple of 2 floats): Minimum and maximum values shown on the
                y-axis. Defaults to (1,-1).
            fps (float): Target plotting rate in frames per second.
                Defaults to 60.
        '''

        super().__init__(*args, **kwargs)

        # should not change after __init__
        self._fs            = fs
        self._lineCount     = lineCount
        self._xLimits       = xLimits
        self._yLimits       = yLimits
        self._yScale        = yScale
        self._yOffset       = yOffset
        self._yGap          = yGap
        self._chunk         = chunk
        self._color         = color if isinstance(color, list) else [color]
        self._fps           = fps

        buffer1Size         = int(self.fs*2)
        self._buffer1       = misc.CircularBuffer((lineCount, buffer1Size))

        self._measuredFPS   = 0
        self._updateLast    = None
        self._updateTimes   = []    # keep last update times for measuring FPS

        self._timer         = QtCore.QTimer()
        self._timer.timeout.connect(self.update)
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
        self.showGrid(x=True, y=True, alpha=0.5)

        # draw all axis as black (draw box)
        self.getAxis('left'  ).setPen(pg.mkPen('k'))
        self.getAxis('bottom').setPen(pg.mkPen('k'))
        self.getAxis('right' ).setPen(pg.mkPen('k'))
        self.getAxis('top'   ).setPen(pg.mkPen('k'))
        self.getAxis('bottom').setTicks([])
        self.getAxis('right' ).setTicks([])
        self.getAxis('right' ).show()
        self.getAxis('top'   ).show()

        # set axis labels
        self.getAxis('top'   ).setLabel('<br />Frequency (kHz)')

        # prepare plots
        self._curves = [None]*self.lineCount
        for i in range(self.lineCount):
            self._curves[i] = self.plot([], [],
                pen=self._color[i % len(self._color)])

    def start(self):
        if not self._timer.isActive():
            self._updateLast = dt.datetime.now()
            self._timer.start()

    def stop(self):
        if self._timer.isActive():
            self._timer.stop()
            self.update()
            self._updateLast = None

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

        # keep a local copy of data in a circular buffer
        # for online processing and fast plotting
        with self._buffer1:
            self._buffer1.write(data)

    def update(self):
        try:
            # log.debug('Updating plot')

            # measure the actual fps
            if self._updateLast is not None:
                updateTime = (dt.datetime.now() -
                    self._updateLast).total_seconds()
                self._updateTimes.append(updateTime)
                if len(self._updateTimes)>60:
                    self._updateTimes.pop(0)
                updateMean = np.mean(self._updateTimes)
                fps = 1/updateMean if updateMean else 0
                self._measuredFPS = fps
            self._updateLast = dt.datetime.now()

            if not self._buffer1.updated: return
            # if self._buffer1.nsAvailable<self._chunk*self.fs: return

            with self._buffer1: # self._refreshLock
                to = self._buffer1.nsWritten
                frm = to-int(self._chunk*self.fs)
                if frm < 0: return
                data = self._buffer1.read(frm, to)
                freq = np.fft.fftfreq(data.shape[-1], 1/self.fs)
                mask = (self._xLimits[0]<=freq) & (freq<=self._xLimits[1])
                freq = freq[mask]
                for line in range(self.lineCount):
                    ps = np.abs(np.fft.fft(data[line,:]))**2 / data.shape[-1]
                    ps = ps[mask]
                    self._curves[line].setData(freq, ps*self._yScale +
                        self._yOffset + line*self._yGap)

        except:
            log.exception('Failed to update FFT plot, stopping plot timer')
            self._timer.stop()


class BaseChannel():
    '''Base class for all channels.

    Contains not much functional code, mostly for class hierarchy purposes.
    '''

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

    def __init__(self, source=None):
        self.source       = source
        self._sink        = None


class AnalogChannel(BaseChannel):
    '''All-in-one class for storage and plotting of multiline analog signals.

    Utilizes the `hdf5` module for storage in HDF5 file format.

    Needs to be paired with a ScrollingPlotWidget for plotting.
    '''

    @property
    def plotWidget(self):
        return self._plotWidget

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
    def lineCount(self):
        return self._lineCount

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

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        # verify
        if hasattr(self, '_threshold') and self._threshold == value:
            return

        with self._refreshLock:
            # save
            self._threshold = value

            # apply
            if not self._refreshBlock:
                self._refresh()

    @property
    def refreshBlock(self):
        return self._refreshBlock

    @refreshBlock.setter
    def refreshBlock(self, value):
        self._refreshBlock = value

    def __init__(self, fs, plotWidget=None, hdf5Node=None, source=None,
            label=None, labelOffset=0, lineCount=1, yScale=1, yOffset=0,
            yGap=1, color=list('rgbcmyk'), chunkSize=.5, filter=(None,None),
            fsPlot=5e3, grandMean=False, threshold=0):
        '''
        Args:
            fs (float): Actual sampling frequency of the channel in Hz.
            plotWidget (ScrollingPlotWidget): Defaults to None.
            hdf5Node (str): Path of the node to store data in HDF5 file.
                Defaults to None.
            source (BaseChannel): ...
            label (str or list of str): ...
            labelOffset (float): ...
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

        super().__init__(source)

        self._refreshBlock   = True
        self._refreshLock    = threading.Lock()

        self._fs             = fs
        self._plotWidget     = plotWidget
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
        self.threshold       = threshold

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

        if hdf5Node is not None:
            if hdf5.contains(hdf5Node):
                raise NameError('HDF5 node %s already exists' % hdf5Node)
            hdf5.createEArray(hdf5Node, tb.Float32Atom(), (0,lineCount),
                '', hdf5Filters, expectedrows=fs*60*30)    # 30 minutes
            hdf5.setNodeAttr(hdf5Node, 'fs', fs)

        if plotWidget:
            labelOffset += np.arange(lineCount)*yGap+yOffset
            plotWidget.add(self, label, labelOffset)

        # threshold guide lines
        pen = pg.mkPen((255,0,0,150), style=QtCore.Qt.DashLine, cosmetic=True)
        self._thresholdGuides = [[None]*2 for i in range(self.lineCount)]
        for i in range(self.lineCount):
            for j in range(2):
                y = (self._yOffset + i*self._yGap +
                    self.yScale*self.threshold*(j*2-1))
                line = pg.InfiniteLine(y, 0, pen=pen)
                line.setVisible(bool(self.threshold))
                self.plotWidget.addItem(line)
                self._thresholdGuides[i][j] = line

        # prepare plotting chunks
        self._chunkCount = int(np.ceil(self.plotWidget.xRange/self._chunkSize))
        self._curves = [[None]*self._chunkCount for i in range(self.lineCount)]
        for i in range(self.lineCount):
            for j in range(self._chunkCount):
                self._curves[i][j] = self.plotWidget.plot([], [],
                    pen=self._color[i % len(self._color)])
                self._curves[i][j].setVisible(self.linesVisible[i])

        self._refreshBlock   = False
        self._refresh()

        self._processThread = threading.Thread(target=self._processLoop)
        self._processThread.daemon = True
        self._processThread.start()

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
        if hasattr(self, '_thresholdGuides'):
            for i in range(self.lineCount):
                for j in range(2):
                    y = (self._yOffset + i*self._yGap +
                        self.yScale*self.threshold*(j*2-1))
                    line = self._thresholdGuides[i][j]
                    line.setValue(y)
                    line.setVisible(bool(self._threshold))

        # reset buffers to the beginning of the current plot window
        nsRange = int(self.plotWidget.xRange*self.fs)
        self._buffer1.nsRead    = self._buffer1.nsRead    // nsRange * nsRange
        nsRange = int(self.plotWidget.xRange*self.fsPlot)
        self._buffer2.nsWritten = self._buffer2.nsWritten // nsRange * nsRange
        # self._chunkLast = 0

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

    def update(self, ts=None, tsMin=None, nextWindow=False):
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

            # update threshold guide lines
            # if self.threshold:
            #     buffer2Size = self._buffer2.shape[self._buffer2.axis]
            #     frm = max(0, self._buffer2.nsWritten-buffer2Size)
            #     data = self._buffer2.read(frm=frm, advance=False)
            #     if data.shape[-1]:
            #         for i in range(self.lineCount):
            #             sigma = np.median(np.abs(data[i,:]))
            #             for j in range(2):
            #                 y = (self._yOffset + i*self._yGap +
            #                     self.yScale*self.threshold*sigma*(j*2-1))
            #                 self._thresholdGuides[i][j].setValue(y)

    def refresh(self):
        with self._refreshLock:
            self._refresh()


class AnalogPlot(pipeline.Sampled):
    '''All-in-one class for storage and plotting of multiline analog signals.

    Utilizes the `hdf5` module for storage in HDF5 file format.

    Needs to be paired with a ScrollingPlotWidget for plotting.
    '''

    @property
    def plotWidget(self):
        return self._plotWidget

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
    def lineCount(self):
        return self._lineCount

    @property
    def yScale(self):
        return self._yScale

    @yScale.setter
    def yScale(self, value):
        self._yScale = value

    def __init__(self, plotWidget, label=None, labelOffset=0,
            yScale=1, yOffset=0, yGap=1, color=list('rgbcmyk'), chunkSize=.5):

        self._plotWidget     = plotWidget
        self._label          = label
        self._labelOffset    = labelOffset
        self._yScale         = yScale
        self._yOffset        = yOffset
        self._yGap           = yGap
        self._color          = color if isinstance(color, list) else [color]
        self._chunkSize      = chunkSize

        super().__init__()

    def _configured(self):
        super()._configured()
        self._lineCount = self._channels

        buffer1Size   = int(self.plotWidget.xRange*self._fs*1.2)
        self._buffer1 = misc.CircularBuffer((self._lineCount, buffer1Size))

        self._labelOffset += (np.arange(self._lineCount) * self._yGap
            + self._yOffset)
        self._plotWidget.add(self, self._label, self._labelOffset)

        # prepare plotting chunks
        self._chunkCount = int(np.ceil(self.plotWidget.xRange
            / self._chunkSize))
        self._curves = [[None]*self._chunkCount
            for i in range(self._lineCount)]
        for i in range(self._lineCount):
            for j in range(self._chunkCount):
                self._curves[i][j] = self.plotWidget.plot([], [],
                    pen=self._color[i % len(self._color)])

    def write(self, data, source=None):
        '''
        Args:
            data (numpy.array): 1D for single line or 2D for multiple lines
                with the following format: lines x samples
        '''
        data = self._verifyData(data)

        # keep a local copy of data in a circular buffer
        # for online processing and fast plotting
        with self._buffer1:
            self._buffer1.write(data)

        super().write(data, source)

    def update(self, ts=None, tsMin=None, nextWindow=False):
        if not self._buffer1.updated: return

        with self._buffer1:
            nsRange      = int(self.plotWidget.xRange*self._fs)
            chunkSamples = int(self._chunkSize*self._fs)

            nsFrom       = self._buffer1.nsRead
            nsFromMin    = nsFrom // nsRange * nsRange
            chunkFrom    = (nsFrom - nsFromMin) // chunkSamples

            nsTo         = self._buffer1.nsWritten
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
                                   nsChunkTo) / self._fs
                    data         = self._buffer1.read(nsChunkFrom, nsChunkTo)
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


class MultiLine(pg.QtGui.QGraphicsPathItem):
    def __init__(self, pen):
        super().__init__()
        self.setPen(pg.mkPen(pen))
    def shape(self):
        return pg.QtGui.QGraphicsItem.shape(self)
    def boundingRect(self):
        return self.path().boundingRect()

    def setData(self, x, y):
        connect = np.ones(x.shape, dtype=bool)
        connect[:,-1] = 0 # don't draw the segment between each trace
        path = pg.arrayToQPath(x.flatten(), y.flatten(), connect.flatten())
        self.setPath(path)


class AnalogChannelFast(AnalogChannel):
    @property
    def linesVisible(self):
        return super().linesVisible

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
                for curve in self._curves:
                    curve.setVisible(value[0])

    def __init__(self, fs, plotWidget=None, hdf5Node=None, source=None,
            label=None, labelOffset=0, lineCount=1, yScale=1, yOffset=0,
            yGap=1, color=list('rgbcmyk'), chunkSize=.5, filter=(None,None),
            fsPlot=5e3, grandMean=False, threshold=0):
        '''
        Args:
            fs (float): Actual sampling frequency of the channel in Hz.
            plotWidget (ScrollingPlotWidget): Defaults to None.
            hdf5Node (str): Path of the node to store data in HDF5 file.
                Defaults to None.
            source (BaseChannel): ...
            label (str or list of str): ...
            labelOffset (float): ...
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

        BaseChannel.__init__(self, source)

        self._refreshBlock   = True
        self._refreshLock    = threading.Lock()

        self._fs             = fs
        self._plotWidget     = plotWidget
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
        self.threshold       = threshold

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

        if hdf5Node is not None:
            if hdf5.contains(hdf5Node):
                raise NameError('HDF5 node %s already exists' % hdf5Node)
            hdf5.createEArray(hdf5Node, tb.Float32Atom(), (0,lineCount),
                '', hdf5Filters, expectedrows=fs*60*30)    # 30 minutes
            hdf5.setNodeAttr(hdf5Node, 'fs', fs)

        if plotWidget:
            labelOffset += np.arange(lineCount)*yGap+yOffset
            plotWidget.add(self, label, labelOffset)

        # threshold guide lines
        pen = pg.mkPen((255,0,0,150), style=QtCore.Qt.DashLine, cosmetic=True)
        self._thresholdGuides = [[None]*2 for i in range(self.lineCount)]
        for i in range(self.lineCount):
            for j in range(2):
                y = (self._yOffset + i*self._yGap +
                    self.yScale*self.threshold*(j*2-1))
                line = pg.InfiniteLine(y, 0, pen=pen)
                line.setVisible(bool(self.threshold))
                self.plotWidget.addItem(line)
                self._thresholdGuides[i][j] = line

        # prepare plotting chunks
        self._chunkCount = int(np.ceil(self.plotWidget.xRange/self._chunkSize))
        self._curves = [None]*self._chunkCount
        for j in range(self._chunkCount):
            self._curves[j] = MultiLine(self._color[0 % len(self._color)])
            self._curves[j].setVisible(self.linesVisible[0])
            self.plotWidget.addItem(self._curves[j])

        self._refreshBlock   = False
        self._refresh()

        self._processThread = threading.Thread(target=self._processLoop)
        self._processThread.daemon = True
        self._processThread.start()

    def update(self, ts=None, tsMin=None, nextWindow=False):
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
                    time = time[np.newaxis,:].repeat(self.lineCount, axis=0)
                    lines = np.arange(self.lineCount)[:,np.newaxis]
                    lines = lines.repeat(data.shape[-1], axis=1)*self._yGap
                    self._curves[chunk].setData(time,
                        data*self._yScale + self._yOffset + lines)
            except:
                log.info('nsRange %d, chunkSamples %d, nsFrom %d, '
                    'nsFromMin %d, chunkFrom %d, nsTo %d, nsToMin %d, '
                    'chunkTo %d, chunk %d', nsRange, chunkSamples,
                    nsFrom, nsFromMin, chunkFrom, nsTo, nsToMin, chunkTo,
                    chunk)
                raise

            # update threshold guide lines
            # if self.threshold:
            #     buffer2Size = self._buffer2.shape[self._buffer2.axis]
            #     frm = max(0, self._buffer2.nsWritten-buffer2Size)
            #     data = self._buffer2.read(frm=frm, advance=False)
            #     if data.shape[-1]:
            #         for i in range(self.lineCount):
            #             sigma = np.median(np.abs(data[i,:]))
            #             for j in range(2):
            #                 y = (self._yOffset + i*self._yGap +
            #                     self.yScale*self.threshold*sigma*(j*2-1))
            #                 self._thresholdGuides[i][j].setValue(y)


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

    @property
    def plotWidget(self):
        return self._plotWidget

    def __init__(self, plotWidget=None, hdf5Node=None, source=None,
            label=None, labelOffset=0):

        super().__init__(source)

        self._plotWidget = plotWidget
        self._hdf5Node   = hdf5Node
        self._data       = []
        self._dataAdded  = []
        self._lock       = threading.Lock()

        if hdf5Node is not None:
            if hdf5.contains(hdf5Node):
                raise NameError('HDF5 node %s already exists' % hdf5Node)
            hdf5.createEArray(hdf5Node, tb.Float64Atom(), (0,2), '',
                hdf5Filters, expectedrows=200)

        if plotWidget:
            plotWidget.add(self, label, labelOffset)

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

    def __init__(self, plotWidget=None, hdf5Node=None, source=None, label=None,
            labelOffset=0, yOffset=0, color=(0,255,0), symbolStart='t1',
            symbolStop='t', symbolSize=15):

        labelOffset += yOffset
        super().__init__(plotWidget, hdf5Node, source, label, labelOffset)

        self._yOffset      = yOffset

        # prepare scatter plots, one for epoch start and one for epoch stop
        kwargs = dict(pen=None, symbolPen=None, symbolBrush=pg.mkBrush(color),
            symbolSize=symbolSize)
        self._scatterStart = plotWidget.plot([], symbol=symbolStart, **kwargs)
        self._scatterStop  = plotWidget.plot([], symbol=symbolStop , **kwargs)

    def update(self, ts=None, tsMin=None, nextWindow=False):
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

    def __init__(self, plotWidget=None, hdf5Node=None, source=None, label=None,
            labelOffset=0, yOffset=0, yRange=1, color=(0,0,255,100)):
        labelOffset += yOffset + yRange/2
        super().__init__(plotWidget, hdf5Node, source, label, labelOffset)

        self._yOffset = yOffset
        self._yRange  = yRange
        self._color   = color

        self._rects   = []

    def update(self, ts=None, tsMin=None, nextWindow=False):
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
