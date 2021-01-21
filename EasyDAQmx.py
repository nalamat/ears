'''Simplified OO interface with NI DAQ devices using PyDAQmx package.


This file is part of the EARS project: https://github.com/nalamat/ears
Copyright (C) 2017-2021 Nima Alamatsaz <nima.alamatsaz@gmail.com>
Copyright (C) 2017-2021 NESH Lab <https://ihlefeldlab.com>
Distributed under GNU GPLv3. See LICENSE.txt for more info.
'''

'''
NI DAQmx API Notes:

Analog output in continuous samples (DAQmx_Val_ContSamps) and no regeneration
(DAQmx_Val_DoNotAllowRegen) mode:
    On a physical device in no regeneration mode when generation of samples
        reaches end of the buffer, the task fails and performing any
        operation on the task later on throws an error. However, this will
        not happen when using a simulated device (set up in NI MAX).
    Last argument of DAQmxCfgSampClkTiming, sampsPerChanToAcquire doesn't really
        set the buffer size. With all values that I provided the buffer size
        remained 0. You can either use DAQmxSetBufOutputBufSize to set buffer
        size explicitly, or write some data before starting the task to set
        buffer size implicitly to the data size.
    DAQmxGetWriteCurrWritePos returns the index of last sample written in buffer
        plus one. Everytime buffer is overwritten by DAQmxWriteAnalogF64, this
        value changes. Even if the new data are less than the size of previous
        write operation, old data in the tail of buffer are thrown away.
    DAQmxWriteAnalogF64 writes data relative to the position and offset set by
        DAQmxSetWriteRelativeTo and DAQmxSetWriteOffset. If the data being
        written are less than the current buffer size, remaining data after the
        last newly written sample will be thrown away. This will also affect
        DAQmxGetWriteCurrWritePos. When writing data at a position in buffer
        where samples are already generated (outputted), an error in thrown.
        This error doesn't stop the write operation all together, meaning the
        part of data which happen to fall after the last generated sample will
        still be written to buffer, and DAQmxGetWriteCurrWritePos will always
        point to the index of last sample written (even if unsuccessful) + 1.
Everything related to pause trigger and software trigger:
    SetDigLvlPauseTrigSrc
    SetDigLvlPauseTrigWhen
    SetPauseTrigType
    GetPauseTrigTerm
    SetDigLvlPauseTrigDigSyncEnable
    SendSoftwareTrigger
    SetExportedAdvTrigOutputTerm
    SetExportedAdvTrigPulseWidth
    ai/PauseTrigger
'''

import re
import sys
import time
import ctypes
import logging
import threading
import numpy     as np
import datetime  as dt

import misc
import pypeline


SIM = '--sim' in sys.argv
log = logging.getLogger(__name__)

if not SIM:
    import PyDAQmx as mx


class BaseTask():
    @property
    def lines(self):
        '''The original `lines` passed to task during initialization.'''
        return self._lines

    @property
    def lineList(self):
        '''Lines as a list of strings.'''
        return self._lineList

    @property
    def lineCount(self):
        '''Number of line in the task.'''
        return self._lineCount

    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return self.__class__.__name__

    @property
    def lock(self):
        return self._lock

    @property
    def running(self):
        if SIM:
            return self._running
        else:
            p_bool32 = ctypes.c_uint32()
            self._task.GetTaskComplete(p_bool32)
            return not p_bool32.value

    def __init__(self, lines, name=None):
        self._name = name
        log.debug('Initializing %s task', self.name)

        if SIM:
            self._running = False
        else:
            self._task = mx.Task(name)

        if isinstance(lines, list):
            self._lines = ','.join(lines)
        else:
            self._lines =  lines
        self._lock      = threading.Lock()

    def _postInit(self):
        '''Do some housekeeping after task initialization.

        This function should be called by child classes at the end of their
        __init__ function.
        '''

        if SIM:
            lineList = []
            for line in self._lines.split(','):
                # strip white space
                line = line.strip()
                # break '/dev1/port0/line0:3' into ['/dev1/port0/line','0','3']
                match = re.match('(.*?)(\d+)(?::(\d+))?$', line)
                if not match: raise ValueError('Invalid line %s for %s task' %
                    (line, self.name))
                tokens = list(match.groups())
                # if no second number, set it the same as first number
                if tokens[2] is None: tokens[2] = tokens[1]
                # convert strings to numbers
                tokens[1] = int(tokens[1])
                tokens[2] = int(tokens[2])
                # for each number, add a line
                for i in range(tokens[1], tokens[2]+1):
                    lineList += [tokens[0]+str(i)]
            self._lineList  = lineList
            self._lineCount = len(lineList)

        else:
            self._task.TaskControl(mx.DAQmx_Val_Task_Commit)

            lineList = ctypes.create_string_buffer(1024)
            self._task.GetTaskChannels(lineList, len(lineList))
            self._lineList = [line.strip() for line in
                str(lineList.value).split(',')]
            self._lineCount = len(self._lineList)

        log.debug('Initialized %s task', self.name)

    def _start(self):
        log.debug('Starting %s task', self.name)
        if SIM:
            self._running = True
        else:
            self._task.StartTask()

    def _stop(self):
        log.debug('Stopping %s task', self.name)
        if SIM:
            self._running = False
            self._stopped()
        else:
            self._task.StopTask()
            self._stopped()
            # reset task buffers and resources to enable restart
            self._task.TaskControl(mx.DAQmx_Val_Task_Unreserve)
            self._task.TaskControl(mx.DAQmx_Val_Task_Commit)

    def _stopped(self):
        '''Called after stopping the task and before reseting it.'''
        pass

    def _clear(self):
        if self.running:
            self._stop()
        log.debug('Clearing %s task', self.name)
        if not SIM:
            self._task.ClearTask()

    def start(self):
        '''Start the task.'''
        with self.lock:
            self._start()

    def stop(self):
        '''Stop and reset the task so it can be restarted.'''
        with self.lock:
            self._stop()

    def clear(self):
        '''Stop and clear all task resources.'''
        with self.lock:
            self._clear()


class DigitalInput(BaseTask, pypeline.Node):
    @property
    def lineNames(self):
        return self._lineNames

    @property
    def edgeDetected(self):
        return self._edgeDetected

    def __init__(self, lines, name=None,
            edgeDetected=None, getTS=None, lineNames=None,
            initialState=0, debounce=5.12e-3):
        '''
        Args:
            lines (str): Example: '/dev1/port0/line1' or /dev2/port1/line0:7'
            name (str): Optional name for the task.
            edgeDetected (callable): ...
            getTS (callable): Get timestamp
            lineNames (list of str): ...
            initialState (int or bool): ...
            debounce (float): Available values: 0, 160e-9, 10.24e-6, 5.12e-3
        '''
        super().__init__(lines, name)

        self._edgeDetected = misc.Event(edgeDetected)
        self._getTS        = getTS
        if isinstance(lineNames, str): lineNames = [lineNames]
        self._lineNames    = lineNames
        self._initialState = initialState

        if not SIM:
            self._task.CreateDIChan(self.lines, '', mx.DAQmx_Val_ChanPerLine)
            self._task.CfgChangeDetectionTiming(self.lines, self.lines,
                mx.DAQmx_Val_ContSamps, 100)
            self._task.CfgInputBuffer(0)

            if debounce:
                self._task.SetDIDigFltrMinPulseWidth(lines, debounce)
                self._task.SetDIDigFltrEnable(lines, True)

            # callback pointer is stored such to prevent being garbage collected
            self._callbackPtr = mx.DAQmxSignalEventCallbackPtr(self._callback)
            self._task.RegisterSignalEvent(mx.DAQmx_Val_ChangeDetectionEvent, 0,
                self._callbackPtr, None)

        super()._postInit()

        pypeline.Node.__init__(self)

        if lineNames:
            if len(lineNames) != self.lineCount:
                raise ValueError('When specified, length of `lineNames` '
                    'should match the actual number of `lines`')

    def _configuring(self, params, sinkParams):
        super()._configuring(params, sinkParams)

        if not self._getTs:
            raise ValueError('Can\'t use in pypeline without a getTS function')

    def _start(self):
        self._lastData = np.full(self.lineCount, self._initialState,
            dtype=np.uint8)

        if SIM:
            self._currentData = self._lastData.copy()

        super()._start()

    def _callback(self, *args):
        '''Called when change is detected on any of the digital inputs.'''
        try:
            with self.lock:
                # if self.running:
                log.debug('%s task change detection callback', self.name)

                if self._getTS:
                    ts = self._getTS()
                else:
                    ts = None

                data = self._read()

                for i in range(len(data)):
                    if data[i] != self._lastData[i]:
                        name = self.lineNames[i] if self.lineNames else i
                        edge = 'rising' if data[i] else 'falling'

                        if ts is None:
                            log.debug('%s task detected %s edge on %s',
                                self.name, edge, name)
                            self.edgeDetected(self, name, edge)
                        else:
                            log.debug('%s task detected %s edge on %s at '
                                '%.3f', self.name, edge, name, ts)
                            self.edgeDetected(self, name, edge, ts)

                # pass data down into the pipeline
                # timestamp for the changed lines, an empty list for others
                data2 = [ts if data[i] != self._lastData[i] else []
                    for i in range(len(data))]
                pypeline.Node.write(self, data2)

                self._lastData = data
                return 0

        except:
            log.exception('Error in %s task change detection callback',
                self.name)
            return -1

    def _read(self):
        samples = 1
        log.debug('Reading %dx%d samples from %s task', self.lineCount,
            samples, self.name)

        if SIM:
            data = self._currentData

        else:
            data = np.empty(self.lineCount, dtype=np.uint8)
            sampsPerChanRead = ctypes.c_int32()
            numBytesPerSamp  = ctypes.c_int32()
            self._task.ReadDigitalLines(samples, 0, mx.DAQmx_Val_GroupByChannel,
                data, data.size, sampsPerChanRead, numBytesPerSamp, None)

        return data

    def read(self):
        '''Return last read state of digital inputs.'''
        with self.lock:
            # if there's only a single value (single line), unwrap and return
            if len(self._lastData)==1:
                return self._lastData[0]
            # otherwise return (a copy of) data in array format
            else:
                return self._lastData.copy()

    def write(self, data):
        '''Simulated software write to the digital input task.

        Only implemented for the `SIM` mode.
        '''

        if not SIM:
            raise NotImplementedError('`write` function only implemented '
                'for `SIM` mode')

        if not self.running:
            raise RuntimeError('%s task is not started yet' % self.name)

        # wrap a single value in a list
        if not isinstance(data, (list, np.ndarray)):
            data = [data]

        # convert to a numpy byte array
        data = np.array(data).astype(np.uint8)

        # check for dimensions size
        if data.shape != self._lastData.shape:
            raise ValueError('Given `data` size should be %s for %s task' %
                (self.name, self._lastData.shape) )

        # update current data
        self._currentData = data

        # simulated change detection callback
        if (self._currentData != self._lastData).any():
            self._callback()


class DigitalOutput(BaseTask):
    def __init__(self, lines, name=None, initialState=None):
        '''
        Args:
            lines (str): Example: '/dev1/port0/line1' or /dev2/port1/line0:7'
            name (str): Optional name for the task.
            initialState (numeric, list or numpy.array): ...
        '''
        super().__init__(lines, name)

        self._initialState = initialState

        if not SIM:
            self._task.CreateDOChan(self.lines, None,
                mx.DAQmx_Val_ChanForAllLines)

        super()._postInit()

    def _start(self):
        if self._initialState is not None:
            self._write(self._initialState)

        super()._start()

    def _write(self, data):
        # wrap a single value in a list
        if not isinstance(data, (list, np.ndarray)):
            data = [data]

        # convert to a numpy byte array
        data = np.array(data).astype(np.uint8)

        # check for dimensions
        if data.ndim != 1:
            raise ValueError('Only a 1-dimensional array can be written to %s '
                'task' % self.name)

        # check for array size
        if len(data) != self.lineCount:
            raise ValueError('The `data` given to %s task should be the same '
                'length as line count' % self.name)

        # write!
        log.debug('Writing %s to %s task', data, self.name)
        if not SIM:
            sampsPerChanWritten = ctypes.c_int32()
            self._task.WriteDigitalLines(1, True, 0,
                mx.DAQmx_Val_GroupByChannel, data, sampsPerChanWritten, None)

    def write(self, data):
        '''Write to digital output lines.

        Args:
            data (numeric, list or numpy.array): A single numeric value or a 1D
                array the same length as line count, wherein 0 and 1
                indicate digital low or high states respectively.
        '''
        with self.lock:
            self._write(data)


class BaseAnalog(BaseTask):
    @property
    def channels(self):
        '''The original `channels` passed to the task during initialization.
        Alias for `lines`.
        '''
        return self.lines

    @property
    def channelList(self):
        '''Channels as a list of strings.
        Alias for `lineList`.
        '''
        return self.lineList

    @property
    def channelCount(self):
        '''Number of channels in the task.
        Alias for `lineCount`.
        '''
        return self.lineCount

    @property
    def fs(self):
        return self._fs

    @property
    def samples(self):
        return self._samples

    def __init__(self, channels, fs, samples, name=None):
        super().__init__(channels, name)

        self._fs              = fs
        self._samples         = samples

        if SIM:
            self._simInterval = 100e-3

    def _postInit(self, accurateFS=True, timebaseSrc=None, timebaseRate=None,
            startTrigger=None):
        '''Do some housekeeping after task initialization.

        This function should be called by child classes at the end of their
        __init__ function.
        '''

        if not SIM:
            fs = ctypes.c_double()
            self._task.GetSampClkRate(fs)
            if self.fs != fs.value:
                if accurateFS:
                    raise ValueError('In %s task given sampling frequency %g '
                        'cannot be accurately generated by the device' %
                        (self.name, self.fs))
                else:
                    log.warning('In %s task sampling frequency %g was generated'
                        'instead of %g', self.name, fs.value, self.fs)
                    self._fs = fs.value

            if timebaseSrc:
                self._task.SetSampClkTimebaseSrc(timebaseSrc)

            if timebaseRate:
                self._task.SetSampClkTimebaseRate(timebaseRate)

            if startTrigger:
                # set the task to start on rising edge of the specified terminal
                self._task.CfgDigEdgeStartTrig(startTrigger,
                    mx.DAQmx_Val_Rising)

        super()._postInit()

    def _start(self):
        if SIM:
            self._simStop          = threading.Event()
            self._simThread        = threading.Thread(target=self._simLoop)
            # self._simThread.daemon = True
            self._simThread.start()

        super()._start()

    def _stop(self):
        if SIM and hasattr(self, '_simThread'):
            self._simStop.set()
            # use a loop to allow keyboard interrupt
            # while self._simThread.isAlive():
            #     self._simThread.join(.1)
            self._simThread.join(1)

        super()._stop()

    def _simLoop(self):
        try:
            log.debug('Started %s task sim thread', self.name)

            tsStart = dt.datetime.now()

            # loop every 0.01 sec, stop the thread if task has been stopped
            while not self._simStop.wait(self._simInterval):
                # make the callback atomic with a lock
                with self.lock:
                    if self.running:
                        ts = (dt.datetime.now() - tsStart).total_seconds()
                        self._simCallback(ts)

            log.debug('Stopped %s task sim thread', self.name)

        except:
            log.exception('Sim thread for %s task failed', self.name)

    def _simCallback(self, ts):
        pass

    def wait(self):
        '''Wait until task is complete.'''
        if self.samples==np.inf:
            raise RuntimeError('Cannot wait for %s task with infinite samples'
                'to complete' % self.name)
        log.debug('Waiting for %s task to complete', self.name)
        while self.running:
            time.sleep(.1)
        log.debug('Wait for %s task completed', self.name)
        # self.WaitUntilTaskDone(mx.DAQmx_Val_WaitInfinitely)


class AnalogInput(BaseAnalog, pypeline.Sampled):
    @property
    def nsRead(self):
        '''Total number of samples read.'''
        if SIM:
            return self._simBuffer.nsRead
        else:
            p_uint64 = ctypes.c_uint64()
            self._task.GetReadCurrReadPos(p_uint64)
            return p_uint64.value

    @property
    def nsAvailable(self):
        '''Number of samples available but not read yet.'''
        if SIM:
            return self._simBuffer.nsAvailable
        else:
            p_uint32 = ctypes.c_uint32()
            self._task.GetReadAvailSampPerChan(p_uint32)
            return p_uint32.value

    @property
    def nsAcquired(self):
        '''Total number of samples acquired.'''
        if SIM:
            return self._simBuffer.nsWritten
        else:
            p_uint64 = ctypes.c_uint64()
            self._task.GetReadTotalSampPerChanAcquired(p_uint64)
            return p_uint64.value

    @property
    def dataAcquired(self):
        '''Data acquired event.

        Callbacks are invoked when a new data chunk has been acquired.
        '''
        return self._dataAcquired

    @property
    def dataChunk(self):
        return self._dataChunk

    def __init__(self, channels, fs, samples, name=None,
            range=(-10.,10.), dataAcquired=None, dataChunk=100e-3,
            bufDuration=30, referenced=True,
            accurateFS=True, timebaseSrc=None, timebaseRate=None,
            startTrigger=None):
        '''
        Args:
            channels (str): Example: '/dev1/ai0' or '/dev2/ai0:15'
            fs (float): Input sampling rate. Depending on the timebase clock,
                device might not be able to generate the requested rate
                accurately.
            samples (int): Number of samples to read from the device. If
                set to numpy.inf, task will go into infinite samples mode.
            name (str): Optional name for the task.
            range (tuple of float): Input voltage range. This adjusts the
                output DAC for the best quantization resolution possible.
            dataAcquired (callable): In infinite samples mode, this function is
                called when a specific number of samples are acquired.
            dataChunk (float): In infinite samples mode, how long in seconds
                until the acquired data are read from the device and given to
                the `dataAcquired` event.
            bufDuration (float): In infinite samples mode, determines how long
                should the software buffer be in seconds. The actual buffer size
                depends on the specified sampling rate.
            referenced (bool): Referenced (AI GND) vs. non-referenced (AI SENSE)
                measurement. See:
                http://www.ni.com/white-paper/3344/en/
                zone.ni.com/reference/en-XX/help/370466Y-01/measfunds/refsingleended/
            accurateFS (bool): If True, when device is not able to generate the
                requested sampling frequency accurately, the task will fail and
                a ValueError will be raised. If False, an inaccurate sampling
                frequency will only be reported with a warning.
            timebaseSrc (str): The terminal to use as the timebase for the
                sample clock. When multiple DAQ devices are connected via an
                RTSI cable, this value can be used to synchronize their sampling
                in order to minimize long term offset caused by clock jitter.
                See the following link for information on terminals:
                zone.ni.com/reference/en-XX/help/370466Y-01/mxcncpts/termnames
            timebaseRate (float): The rate of the specified `timebaseSrc` in Hz.
                This value is only needed when using an external timebase.
            startTrigger (str): Configures the task to begin acquisition
                on rising edge of a digital signal. Note that the task should
                have already been started for the trigger to work.
                See the following link for information on trigger sources:
                zone.ni.com/reference/en-XX/help/370466Y-01/mxcncpts/termnames
        '''
        super().__init__(channels, fs, samples, name)

        if SIM and samples!=np.inf:
            raise NotImplementedError('AnalogInput is not yet implemented for '
                'finite samples in `SIM` mode')

        if not SIM:
            config = mx.DAQmx_Val_RSE if referenced else mx.DAQmx_Val_NRSE
            self._task.CreateAIVoltageChan(self.channels, None, config,
                *range, mx.DAQmx_Val_Volts, None)

        self._dataAcquired = misc.Event(dataAcquired)
        self._dataChunk    = dataChunk

        if samples==np.inf:
            bufSize = int(np.ceil(fs*bufDuration))

            if not SIM:
                self._task.CfgSampClkTiming('', fs, mx.DAQmx_Val_Rising,
                    mx.DAQmx_Val_ContSamps, 0)
                self._task.SetBufInputBufSize(bufSize)

                self._callbackPtr = mx.DAQmxEveryNSamplesEventCallbackPtr(
                    self._callback)
                self._task.RegisterEveryNSamplesEvent(
                    mx.DAQmx_Val_Acquired_Into_Buffer, int(dataChunk*fs), 0,
                    self._callbackPtr, None)

        else:
            bufSize = samples

            if not SIM:
                self._task.CfgSampClkTiming('', fs, mx.DAQmx_Val_Rising,
                    mx.DAQmx_Val_FiniteSamps, samples)

        self._postInit(accurateFS, timebaseSrc, timebaseRate,
            startTrigger)

        pypeline.Sampled.__init__(self, fs=self._fs, channels=self._lineCount)

        if SIM:
            self._simBuffer = misc.CircularBuffer((self.lineCount, bufSize))
            # set simulated acquisition interval
            self._simInterval = dataChunk

    def _start(self):
        # if self._samples==np.inf and not self.dataAcquired:
        #     raise ValueError('Before starting %s task in infinite samples '
        #         'mode, a callback must be specified for `dataAcquired` event' %
        #         self.name)

        super()._start()

    def _stopped(self):
        '''Called after stopping the task and before reseting it.'''
        # super()._stop()
        if SIM:
            self._simBuffer.nsWritten = 0

    def _read(self, ns=None, wait=True):
        if ns==None:
            if self.samples==np.inf:
                ns = self.nsAvailable
            else:
                ns = self.samples

        # log.debug('Reading %dx%d samples from %s task', self.lineCount,
        #     ns, self.name)

        if SIM:
            if wait:
                while self.nsAvailable < ns:
                    self._simBuffer.wait()
            elif self.nsAvailable < ns:
                raise ValueError('%s task doesn\'t have %d samples available' %
                    (self.name, self.nsAvailable) )
            data = self._simBuffer.read(to=self._simBuffer.nsRead+ns)

        else:
            if wait:
                waitTime = mx.DAQmx_Val_WaitInfinitely
            else:
                waitTime = 0

            data = np.empty((self.lineCount, ns))

            sampsPerChanRead = ctypes.c_int32()
            self._task.ReadAnalogF64(ns, waitTime, mx.DAQmx_Val_GroupByChannel,
                data, data.size, sampsPerChanRead, None)

        return data

    def _callback(self, *args):
        '''Called when N samples have been acquired by the task.'''
        try:
            with self.lock:
                if self.running:
                    log.debug('%s task every N samples callback', self.name)
                    self._callDataAcquired()
            return 0

        except:
            log.exception('Error in %s task every N samples callback',
                self.name)
            return -1

    def _simCallback(self, ts):
        if self.samples==np.inf:
            if self.fs*self._dataChunk <= self.nsAvailable:
                self._callDataAcquired()

    def _callDataAcquired(self):
        data = self._read()
        # pass data to all registered callbacks
        if self.dataAcquired:
            self.dataAcquired(self, data)
        # pass data down into the pipeline
        pypeline.Sampled.write(self, data)

    def read(self, ns=None, wait=True):
        '''Read samples from the device.

        Args:
            ns (int): Number of samples to read. If None, for infinite
                samples mode, all samples already available in buffer will be
                read. If None, for finite samples mode, the original number of
                samples given when creating the task will be read.
            wait (bool): Whether to wait for the requested number of samples to
                become available. If False and samples not available, an error
                will be raised.
        '''
        with self.lock:
            return self._read(ns, wait)

    def write(self, data):
        '''Software write to the analog input task.

        Only implemented for the `SIM` mode.
        '''

        if not SIM:
            raise NotImplementedError('`write` function only implemented '
                'for `SIM` mode')

        if not self.running:
            raise RuntimeError('%s task is not started yet' % self.name)

        # convert to a numpy 64-bit float array
        data = np.array(data).astype(np.float64)

        if data.ndim == 1: data = data.reshape((1,-1))

        # check for data line count
        if data.shape[0] != self.lineCount:
            raise ValueError('Size of first dimension of `data` should match '
                '`lineCount` for %s task' % self.name)

        self._simBuffer.write(data)


class AnalogOutput(BaseAnalog):
    @property
    def onbrdBufSize(self):
        if SIM:
            raise NotImplementedError('No `onbrdBufSize` in `SIM` mode')
        else:
            p_uint32 = ctypes.c_uint32()
            self._task.GetBufOutputOnbrdBufSize(p_uint32)
            return p_uint32.value

    @property
    def bufSize(self):
        if SIM:
            return self._simBuffer.shape[self._simBuffer.axis]
        else:
            p_uint32 = ctypes.c_uint32()
            self._task.GetBufOutputBufSize(p_uint32)
            return p_uint32.value

    @property
    def nsWritten(self):
        '''Total number of samples written.'''
        if SIM:
            return self._simBuffer.nsWritten + self._nsOffset
        else:
            p_uint64 = ctypes.c_uint64()
            self._task.GetWriteCurrWritePos(p_uint64)
            return p_uint64.value + self._nsOffset

    @property
    def nsGenerated(self):
        '''Total number of samples generated so far.'''
        if SIM:
            return self._simBuffer.nsRead + self._nsOffset
        else:
            p_uint64 = ctypes.c_uint64()
            self._task.GetWriteTotalSampPerChanGenerated(p_uint64)
            return p_uint64.value + self._nsOffset

    @property
    def nsAvailable(self):
        '''Number of samples available for writing to the output buffer.'''
        if SIM:
            return self._simBuffer.nsWritten - self._simBuffer.nsRead - \
                self._simBuffer.shape[self._simBuffer.axis]
        else:
            p_uint32 = ctypes.c_uint32()
            self._task.GetWriteSpaceAvail(p_uint32)
            return p_uint32.value

    @property
    def dataNeeded(self):
        '''Data needed event.

        Callbacks are invoked before generation reaches end of buffer.
        '''
        return self._dataNeeded

    @property
    def dataChunk(self):
        return self._dataChunk

    def __init__(self, channels, fs, samples, name=None,
            range=(-10.,10.), regenerate=False, dataNeeded=None, dataChunk=5,
            onbrdBufDuration=5e-3, bufDuration=30,
            accurateFS=True, timebaseSrc=None, timebaseRate=None,
            startTrigger=None):
        '''
        Args:
            channels (str): Example: '/dev1/ao0' or '/dev2/ao0:1'
            fs (float): Output sampling rate. Depending on the timebase clock,
                device might not be able to generate the requested rate
                accurately.
            samples (int): Number of samples to output from the device. If
                set to numpy.inf, task will go into infinite samples mode.
            name (str): Optional name for the task.
            range (tuple of float): Output voltage range. This adjusts the
                output DAC for the best quantization resolution possible.
            regenerate (bool): If True, task will output the indicated number of
                samples periodically until stopped.
            dataNeeded (callable): In infinite samples mode, this function is
                called before starting the task with an empty buffer or before
                sample generation reaches end of the buffer.
            dataChunk (float): In infinite samples mode, how long in seconds
                should be left to the end of buffer until the `dataNeeded`
                event is called.
            onbrdBufDuration (float): In infinite samples mode, deteremines how
                long should the onboard buffer be in seconds. This value affects
                the speed at which samples are transferred from software buffer
                to onboard buffer and is crucial when fast modifications to the
                buffer are needed. However, this only matters when using a PCIe
                DAQ model and not for a USB model. The actual buffer size
                depends on the specified sampling rate and will be clipped
                between 2 and 8191 samples.
            bufDuration (float): In infinite samples mode, determines how long
                should the software buffer be in seconds. The actual buffer size
                depends on the specified sampling rate.
            accurateFS (bool): If True, when device is not able to generate the
                requested sampling frequency accurately, the task will fail and
                a ValueError will be raised. If False, an inaccurate sampling
                frequency will only be reported with a warning.
            timebaseSrc (str): The terminal to use as the timebase for the
                sample clock. When multiple DAQ devices are connected via an
                RTSI cable, this value can be used to synchronize their sampling
                in order to minimize long term offset caused by clock jitter.
                See the following link for information on terminals:
                zone.ni.com/reference/en-XX/help/370466Y-01/mxcncpts/termnames
            timebaseRate (float): The rate of the specified `timebaseSrc` in Hz.
                This value is only needed when using an external timebase.
            startTrigger (str): Configures the task to begin output generation
                on rising edge of a digital signal. Note that the task should
                have already been started for the trigger to work.
                See the following link for information on terminals:
                zone.ni.com/reference/en-XX/help/370466Y-01/mxcncpts/termnames
        '''
        super().__init__(channels, fs, samples, name)

        if samples==np.inf and regenerate:
            raise ValueError('In %s task, cannot regenerate infinite number of '
                'samples' % self.name)

        if SIM and samples!=np.inf:
            raise NotImplementedError('AnalogOutput is not yet implemented for '
                'finite samples in `SIM` mode')

        if not SIM:
            self._task.CreateAOVoltageChan(self.channels, '', *range,
                mx.DAQmx_Val_Volts, '')

        self._nsOffset   = 0
        self._regenerate = regenerate
        self._dataNeeded = misc.Event(dataNeeded, singleCallback=True)
        self._dataChunk  = dataChunk

        if samples==np.inf:
            bufSize = int(np.ceil(fs*bufDuration))

            if not SIM:
                self._task.CfgSampClkTiming('', fs, mx.DAQmx_Val_Rising,
                    mx.DAQmx_Val_ContSamps, 0)
                self._task.SetWriteRegenMode(mx.DAQmx_Val_DoNotAllowRegen)
                onbrdBufSize = int(np.clip(np.ceil(fs*onbrdBufDuration),
                    2, 8191))
                self._task.SetBufOutputOnbrdBufSize(onbrdBufSize)

                self._task.SetBufOutputBufSize(bufSize)

                # this callback acts more like a timer, it doesn't matter after
                # how many samples is it called, it just has to happen before
                # sample generation reaches the end of buffer
                self._callbackPtr = mx.DAQmxEveryNSamplesEventCallbackPtr(
                    self._callback)
                self._task.RegisterEveryNSamplesEvent(
                    mx.DAQmx_Val_Transferred_From_Buffer, int(dataChunk*fs/2),
                    0, self._callbackPtr, None)

        elif regenerate:
            self._task.CfgSampClkTiming('', fs, mx.DAQmx_Val_Rising,
                mx.DAQmx_Val_ContSamps, samples)
            self._task.SetWriteRegenMode(mx.DAQmx_Val_AllowRegen)

        else:
            self._task.CfgSampClkTiming('', fs, mx.DAQmx_Val_Rising,
                mx.DAQmx_Val_FiniteSamps, samples)

        self._postInit(accurateFS, timebaseSrc, timebaseRate, startTrigger)

        if SIM:
            self._simBuffer   = misc.CircularBuffer((self.lineCount, bufSize))
            self._simInterval = 10e-3    # set simulated generation interval
            self._simSink     = None
            self._simSinkMap  = None

    def _start(self):
        if self._samples==np.inf and not self.dataNeeded:
            raise ValueError('Before starting %s task in infinite samples '
                'mode, a callback must be specified for `dataNeeded` event' %
                self.name)

        # for infinite samples mode, initialize output buffer before starting
        # the task by calling the `dataNeeded` event
        if (self.samples==np.inf and
                self.nsWritten-self.nsGenerated < self.fs*self._dataChunk):
            self._callDataNeeded()

        super()._start()

    def _stopped(self):
        '''Called after stopping the task and before reseting it.'''
        # log.info('`_stopping` for %s task', self.name)
        if self._samples == np.inf:
            self._nsOffset = self.nsGenerated
        if SIM:
            self._simBuffer.nsWritten = 0

    def _write(self, data, at=None):
        data = np.array(data).astype(np.float64)
        if data.ndim==1: data.reshape((1,len(data)))

        if at==None or at<0:
            at = 0 if at==None else at
            if not SIM:
                self._task.SetWriteRelativeTo(mx.DAQmx_Val_CurrWritePos)
                self._task.SetWriteOffset(at)
            at = self.nsWritten - at
        else:
            if not SIM:
                self._task.SetWriteRelativeTo(mx.DAQmx_Val_FirstSample)
                self._task.SetWriteOffset(at - self._nsOffset)

        log.debug('Writing to %s task: %d samples at %s (generation at %d)',
            self.name, data.shape[-1], at,
            self.nsGenerated if self.running else 0)

        if SIM:
            self._simBuffer.write(data, at - self._nsOffset)
        else:
            sampsPerChanWritten = ctypes.c_int32()
            self._task.WriteAnalogF64(data.shape[-1], False, 0,
                mx.DAQmx_Val_GroupByChannel, data, sampsPerChanWritten, None)

    def _callback(self, *args):
        '''Called when N samples have been generated by the task.'''
        try:
            with self.lock:
                if self.running:
                    log.debug('%s task every N samples callback', self.name)
                    if (self.nsWritten-self.nsGenerated <
                            self.fs*self._dataChunk):
                        self._callDataNeeded()
            return 0

        except:
            log.exception('Error in %s task every N samples callback',
                self.name)
            return -1

    def _simCallback(self, ts):
        # only in infinite samples mode
        if self.samples==np.inf:
            # if less than the specified duration left to the end of buffer
            if self.nsWritten-self.nsGenerated < self._dataChunk*self.fs:
                self._callDataNeeded()

            to = int(ts*self.fs)
            if self._simSink:
                ds = int(self.fs / self._simSink.fs)
                to = to // ds * ds
                if to <= self._simBuffer.nsRead: return
                # log.debug('Reading %s task sim buffer to %d (nsWritten %d)' %
                #     (self.name, to, self.nsWritten) )
                data = self._simBuffer.read(to=to).copy()
                if self._simSinkMap:
                    data = data[self._simSinkMap,:]
                data = data.reshape(data.shape[0], -1, ds)
                data = data.mean(axis=2)
                with self._simSink.lock:
                    if self._simSink.running:
                        self._simSink.write(data)
            else:
                self._simBuffer.read(to=to)

    def _callDataNeeded(self):
        nsWritten = self.nsWritten# if self.running else 0
        nsGenerated = self.nsGenerated# if self.running else 0
        log.debug('Data needed for %s task at %d (generation at %d)',
            self.name, nsWritten, nsGenerated)
        nsNeeded = int(self.fs*self._dataChunk)
        data = self.dataNeeded(self, nsWritten, nsNeeded)
        if data.shape[-1] < nsNeeded:
            log.warning('Buffer underflow may occur in %s task, %d samples '
                'needed but got %d instead', self.name, nsNeeded, data.shape[-1])
        self._write(data)

    def write(self, data, at=None):
        '''Write data to output buffer for generation.
        Args:
            data (list or numpy.ndarray): Data to output from the device.
            at (int): Where to write given data in the buffer. If None, data
                will be written at the end of buffer. Negative numbers indicate
                an offset from the end of buffer, whereas zero or positive
                numbers show an offset from the beginning of buffer.
        '''
        with self.lock:
            self._write(data, at)
            if self.samples==np.inf:
                if (self.nsWritten-self.nsGenerated <
                        self.fs*self._dataChunk):
                    self._callDataNeeded()

    def connect(self, sink, map=None):
        '''Software connect the analog output to an analog input (i.e. sink).

        Only implemented for the `SIM` mode.

        Args:
            sink (AnalogInput): ...
            map (array-like): ...
        '''

        if not SIM:
            raise NotImplementedError('`connect` function only implemented '
                'for `SIM` mode')

        self._simSink    = sink
        self._simSinkMap = map


if __name__ == '__main__':
    try:
        import matplotlib.pyplot as plt

        logging.basicConfig(format='[%(asctime)s.%(msecs)03d, %(module)s, '
            '%(funcName)s, %(levelname)s] %(message)s', level=logging.DEBUG,
            datefmt='%Y/%m/%d-%H:%M:%S')

        # # finite samples
        # analogInput = AnalogInput('/dev1/ai0', fs=1000, samples=1000)
        # analogInput.start()
        # data = analogInput.read()
        #
        # # infinite samples
        # def process(data):
        #     plot(data)
        #
        # analogInput2 = AnalogInput('/dev1/ai0', fs=1000, samples=np.inf, dataAqcuired=process)
        # analogInput2.start()

        if len(sys.argv) > 1 and sys.argv[1]=='dig':
            input = DigitalInput('/dev3/port0/line0:1',
                lineNames=['poke','spout'])
            input.start()

            while True:
                time.sleep(1)

        else:
            def dataNeeded(task, nsWritten, samples):
                indices = np.arange(nsWritten, nsWritten+samples)
                indices %= outData.shape[-1]    # wrap around the end of array
                return outData[indices]

            def dataAcquired(task, data):
                global inpData
                inpData = np.c_[inpData, data]    # append

            def monitor():
                try:
                    status = ()

                    while True:
                        with out.lock:
                            status1 = ('Out:', out.running,
                                out.nsWritten, out.nsGenerated)
                        with inp.lock:
                            status2 = ('Inp:', inp.running,
                                inp.nsRead, inp.nsAcquired)

                        if status != status1 + status2:
                            print(*status1, '-', *status2)
                            status = status1 + status2
                        time.sleep(.1)
                except:
                    log.exception('Monitor thread failed')

            inpFS = 200e3
            inpData = np.empty((1,0))
            # slave task
            # inp = AnalogInput('/dev2/ai0', inpFS, np.inf,
            #     dataAcquired=dataAcquired, dataChunk=.5,
            #     timebaseSrc='/dev1/20MHzTimebase', timebaseRate=20e6,
            #     startTrigger='/dev1/ao/StartTrigger')
            inp = AnalogInput('/dev3/ai0', inpFS, np.inf,
                dataAcquired=dataAcquired, dataChunk=.5,
                startTrigger='/dev3/ao/StartTrigger')

            outFS = 200e3
            t = np.arange(0,1,1/outFS)
            outData = np.sin(2*np.pi*5*t)
            # master task
            # out = AnalogOutput('/dev1/ao1', outFS, np.inf,
            #     dataNeeded=dataNeeded, timebaseSrc='20MHzTimebase')
            out = AnalogOutput('/dev3/ao0', outFS, np.inf,
                dataNeeded=dataNeeded)

            if SIM:
                out.connect(inp)

            thread = threading.Thread(target=monitor)
            # thread.daemon = True
            thread.start()

            inp.start()
            out.start()
            time.sleep(1.1)
            # out.write(np.zeros(int(out.fs*.5)),
            #     int(out.nsGenerated+out.fs*10e-3))
            time.sleep(1)
            out.stop()
            out.reset()
            time.sleep(1.3)
            out.start()
            time.sleep(2)
            inp.stop()
            out.stop()

            plt.plot(np.arange(0, inpData.shape[-1])/inpFS, inpData[0])
            plt.grid(True)
            plt.show()

    except KeyboardInterrupt:
        pass
    except SystemExit:
        raise
    except:
        log.exception('Main loop failed')
    finally:
        if 'out' in dir():
            del out
        if 'inp' in dir():
            del inp
