'''Pipeline model for routing and online processing of data streams.


This file is part of the EARS project: https://github.com/nalamat/ears
Copyright (C) 2017-2018 Nima Alamatsaz <nima.alamatsaz@njit.edu>
Copyright (C) 2017-2018 Antje Ihlefeld <antje.ihlefeld@njit.edu>
Distrubeted under GNU GPLv3. See LICENSE.txt for more info.
'''

import time
import queue
import logging
import fractions
import threading
import scipy.signal
import numpy        as np
import scipy        as sp

import misc


log = logging.getLogger(__name__)


class Route():
    '''Store input and output nodes of a pipeline route.

    Routes can be connected to other Route(s) using the | or >> operators.

    A route can be thought of as a black box that doesn't contain any
    functional data processing or flow control.
    '''

    @classmethod
    def verify(cls, routes):
        '''Verify the given object(s) as instance(s) of `Route`.

        Return value will always be a tuple of the given routes.
        '''

        # cast to a tuple
        if not hasattr(routes, '__len__'):
            routes = (routes,)
        else:
            routes = tuple(routes)

        # check if empty
        # if not routes:
        #     raise ValueError('`routes` cannot be empty')

        # check the type
        for route in routes:
            if not isinstance(route, Route):
                raise TypeError('`routes` should be instance(s) of `Route`')

        return routes

    @classmethod
    def merge(cls, routes):
        '''Merge inputs and outputs of an iterable of parallel routes.

        Construct and return a new route.
        '''

        # verify route types
        routes = cls.verify(routes)

        # extract inputs and outputs of all routes
        inputs  = tuple()
        outputs = tuple()
        for route in routes:
            inputs  += route._inputs
            outputs += route._outputs

        return Route(inputs, outputs)

    @classmethod
    def connect(cls, sources, sinks):
        ''''Connect outputs of all sources to inputs of all sinks.

        Connection is made in a star-like formation and between every two nodes.
        '''

        # merge inputs and outputs of all sources and sinks
        source  = cls.merge(sources )
        sink    = cls.merge(sinks   )

        # connect source outputs to all sink inputs
        for sourceOutput in source._outputs:
            sourceOutput._addSinks(sink._inputs)

        # connect sink inputs to all source outputs
        for sinkInput in sink._inputs:
            sinkInput._addSources(source._outputs)

        # construct the new route
        return Route(source._inputs, sink._outputs)

    def __init__(self, inputs, outputs):
        self._inputs  = self.verify(inputs)
        self._outputs = self.verify(outputs)

    def __or__(self, sinks):
        return self.connect(self, sinks)

    def __ror__(self, sources):
        return self.connect(sources, self)

    def __rshift__(self, sinks):
        return self.connect(self, sinks)

    def __rrshift__(self, sources):
        return self.connect(sources, self)

    def _addSinks(self, sinks):
        raise NotImplementedError()

    def _addSources(self, sources):
        raise NotImplementedError()


class Node(Route):
    '''Receive data, allow manipulation and pass it downstream.'''

    def __init__(self, **kwargs):
        '''Keyword arguments are passed down as `params` to sinks. Depending on
        child class implementation, nodes may choose to use them, stay
        indifferent, or modify them before passing on to their sinks.

        If keyword arguments are given, node is configured at the end of
        __init__ fucntion. Hence, when overriding in child classes, call super
        function at the end of child initialization to prevent an overwrite.
        '''

        # node is a route which both its input and output are the node itself
        super().__init__(self, self)

        self._sources    = tuple()
        self._sinks      = tuple()
        self._params     = dict()
        self._sinkParams = dict()

        self._config(kwargs)

    def _addSinks(self, sinks):
        # config new sinks
        for sink in sinks:
            sink._config(self._sinkParams)
        # keep track of connected sinks
        self._sinks += sinks

    def _addSources(self, sources):
        # keep track of connected sources
        self._sources += sources

    def _config(self, params, sinkParams=None):
        '''Verify params, save, send to sinks and notify child classes.

        Best not to override.
        '''
        if not params or params == self._params:
            return

        params     = params.copy()
        sinkParams = params.copy() if sinkParams is None else sinkParams.copy()

        # allow child classes to verify and change params
        self._configuring(params, sinkParams)

        # save params and sinkParams
        self._params     = params
        self._sinkParams = sinkParams

        # pass sinkParams to downstream nodes
        for sink in self._sinks:
            sink._config(sinkParams)

        self._configured(params)

    def _configuring(self, params, sinkParams):
        '''Allow child classes to verify or change params before their applied.

        When overriding, best practice is to call super function first unless
        behavior masking is intended, e.g. to allow reconfiguration.
        '''

        if self._params:
            raise RuntimeError('Node can be configured only once')

    def _configured(self, params):
        '''Allow child classes to act on applied param changes.

        Mostly needed for initializing local state based on params.
        When overriding, best practice is to call super function first.
        '''

        pass

    def write(self, data, source=None):
        '''Write a chunk of data to the node and send downstream to sinks.

        Child classes can override to implement data processing. When
        overriding, call super function at the end to pass data to sinks, unless
        behavior masking is intended, e.g. to send different data to each sink.
        '''

        # transparently write data to all sinks
        for sink in self._sinks:
            sink.write(data, self)

    def wait(self):
        '''Wait for asynchronous sinks in the pipeline to finish processing.

        Child classes can override to implement waiting. When overriding, call
        super function at the end to wait for rest of the pipeline to finish.
        '''

        for sink in self._sinks:
            sink.wait()


class DummySink(Node):
    '''Dummy sink can have any number of inputs and but no outputs.'''

    def __init__(self, inputs=1):
        super().__init__()

        self._inputs  = (self,)*inputs
        self._outputs = tuple()


class Print(Node):
    '''Print each given chunk of data.'''

    def write(self, data, source=None):
        print(data)
        super().write(data, source)


class Func(Node):
    '''Apply the specified function to each chunk of data.'''

    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)

        if not callable(func):
            raise TypeError('`func` should be callable for % node')

        self._func = func

    def write(self, data, source=None):
        data = self._func(data)
        super().write(data, source)


class Thread(Node):
    '''Pass data onto sink nodes in a daemon thread.'''

    def __init__(self, autoStart=True, daemon=True, **kwargs):
        super().__init__(**kwargs)

        # self._autoStart  = autoStart
        # self._daemon     = daemon

        # self._thread     = None
        # self._threadStop = threading.Event()

        self._queue         = queue.Queue()
        self._thread        = threading.Thread(target=self._loop)
        self._thread.daemon = True

    def _loop(self):
        # while not self._threadStop.isSet():
        #     try:
        #         data = self._queue.get(timeout=0.1)
        #         super().write(data)
        #     except queue.Queue.Empty:
        #         pass

        while True:
            (data, source) = self._queue.get()
            super().write(data, source)

    # def start(self):
    #     if self._thread is None:
    #         self._thread        = threading.Thread(target=self._loop)
    #         self._thread.daemon = self._daemon
    #
    #     if not self._therad.isAlive():
    #         self._thread.start()

    # def stop(self):
    #     if self._thread is not None and self._thread.isAlive():
    #         self._threadStop.set()
    #         self._thread.join()
    #         self._threadStop.clear()
    #         self._thread = None

    def wait(self):
        # TODO: fix. with current implementation data will have been read from
        # the queue (hence empty queue) but not necessariliy completed
        # processing in downstream nodes
        while not self._queue.empty():
            time.sleep(50e-3)
        super().wait()

    def write(self, data, source=None):
        if not self._thread.isAlive():
            self._thread.start()
        self._queue.put((data, source))


class Sampled(Node):
    '''Specialized node for sampled, multichannel signals.

    Data written to this node is constrained to 1D or 2D signals where the
    first dimension (number of channels) is constant.
    '''

    def __init__(self, **kwargs):
        self._fs       = None
        self._channels = None

        super().__init__(**kwargs)

    def _configuring(self, params, sinkParams):
        super()._configuring(params, sinkParams)

        if 'fs' not in params or 'channels' not in params:
            raise ValueError('Sampled node requires `fs` and `channels`')

    def _configured(self, params):
        super()._configured(params)

        self._fs       = params['fs']
        self._channels = params['channels']

    def _verifyData(self, data):
        if not isinstance(data, np.ndarray): data = np.array(data)
        if data.ndim == 1: data = data[np.newaxis, :]
        if data.ndim != 2: raise ValueError('`data` should be 1D or 2D')

        if self._fs is None or self._channels is None:
            raise RuntimeError('Node is not configured')

        if self._channels != data.shape[0]:
            raise ValueError('`data` channel count does not match `channels`')

        return data


class Split(Sampled):
    '''Split multichannel data into multiple sink nodes.'''

    def _configuring(self, params, sinkParams):
        super()._configuring(params, sinkParams)

        # each sink will only receive 1 channel
        sinkParams['channels'] = 1

    def write(self, data, source=None):
        data = self._verifyData(data)

        if self._channels != len(self._sinks):
            raise ValueError('`channels` should match `sink` count')

        for sink, channel in zip(self._sinks, data):
            sink.write(channel, self)


class LFilter(Sampled):
    '''Causal IIR lowpass, highpass or bandpass filter.'''

    @property
    def fl(self):
        '''Low cutoff frequency'''
        return self._fl

    @fl.setter
    def fl(self, fl):
        self._fl = fl
        self._refresh()

    @property
    def fh(self):
        '''High cutoff frequency'''
        return self._fh

    @fh.setter
    def fh(self, fh):
        self._fh = fh
        self._refresh()

    @property
    def n(self):
        '''Filter order'''
        return self._n

    @n.setter
    def n(self, n):
        self._n  = n
        self._zi = None
        self._refresh()

    def __init__(self, fl=None, fh=None, n=6, **kwargs):
        '''
        Args:
            fl (float): Low cutoff frequency.
            fh (float): High cutoff frequency.
            n (int): Filter order.
        '''

        self._fl        = fl
        self._fh        = fh
        self._n         = n

        self._ba        = None
        self._zi        = None

        super().__init__(**kwargs)

    def _configured(self, params):
        super()._configured(params)

        self._refresh()

    def _refresh(self):
        # called when any of the following changes: fs, channels, fl, fh, n
        if self._fs is None or self._channels is None:
            ba = None
        # elif self._fl is not None and self._fh is not None:
        elif self._fl  and self._fh:
            ba = sp.signal.butter(self._n,
                (self._fl/self._fs*2, self._fh/self._fs*2), 'bandpass')
        # elif self._fl is not None:
        elif self._fl:
            ba = sp.signal.butter(self._n, self._fl/self._fs*2, 'highpass')
        # elif self._fh is not None:
        elif self._fh:
            ba = sp.signal.butter(self._n, self._fh/self._fs*2, 'lowpass')
        else:
            ba = None

        # reset initial filter state
        if ba is not None and np.any(np.array(self._ba) != np.array(ba)):
            self._zi = None

        self._ba = ba

    def write(self, data, source=None):
        data = self._verifyData(data)

        if self._ba is not None:
            # initialize filter in steady state
            if self._zi is None:
                self._zi = sp.signal.lfilter_zi(*self._ba)
                self._zi = np.tile(self._zi, self._channels
                    ).reshape((self._channels, -1))
                self._zi = self._zi * data[:,0].reshape((self._channels, -1))

            # apply the IIR filter
            data, self._zi = sp.signal.lfilter(*self._ba, data, zi=self._zi)

        super().write(data, source)


class Downsample(Sampled):
    def __init__(self, ds, bin=None, **kwargs):
        if ds < 1:          raise ValueError('`ds` should be >= 1')
        if round(ds) != ds: raise ValueError('`ds` should be an integer')

        self._ds     = ds
        self._bin    = ds if bin is None else bin
        self._buffer = None

        super().__init__(**kwargs)

    def _configuring(self, params, sinkParams):
        super()._configuring(params, sinkParams)

        sinkParams['fs'] = params['fs'] / self._ds

    def _configured(self, params):
        super()._configured(params)

        self._buffer = misc.CircularBuffer((self._channels, int(self._fs*10)))

    def _downsample(self, data):
        raise NotImplementedError()

    def write(self, data, source=None):
        data = self._verifyData(data)

        # if ds==1, transparently pass data down the pipeline
        if self._ds != 1:
            # TODO: find a better solution than writing to a buffer
            # all nodes should be able to process arbitrarily large signals
            self._buffer.write(data)
            ns = self._buffer.nsWritten
            ns = ns // self._bin * self._bin
            if ns <= self._buffer.nsRead: return
            data = self._buffer.read(to=ns)

            data = self._downsample(data)

        super().write(data, source)


class DownsampleAverage(Downsample):
    def _downsample(self, data):
        data = data.reshape(data.shape[0], -1, self._ds)
        data = data.mean(axis=2)
        return data


class DownsampleMinMax(Downsample):
    '''Divide data into bins and choose only the min and max of each bin.'''

    def __init__(self, ds, **kwargs):
        super().__init__(ds, bin=ds*2, **kwargs)

    def _downsample(self, data):
        data = data.reshape(self._channels, -1, self._bin)
        data = np.stack((data.min(axis=-1), data.max(axis=-1)), axis=2)
        data = data.reshape(self._channels, -1)
        return data


class DownsampleLTTB(Sampled):
    ''''Downsample using the Largest Triangle Three Buckets (LTTB) algorithm.

    See Steinarsson (2013):
    https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf
    '''

    def __init__(self, fsOut, **kwargs):
        self._fsOut  = fsOut
        self._buffer = None
        self._last   = None

        super().__init__(**kwargs)

    def _configuring(self, params, sinkParams):
        super()._configuring(params, sinkParams)

        sinkParams['fs'] = self._fsOut

    def _configured(self, params):
        super()._configured(params)

        if self._fs < self._fsOut:
            raise ValueError('`fsOut` should be >= `fs`')

        if self._fs != self._fsOut:
            # use the Fraction class to determine the smallest number of input
            # and output samples to yield the required downsampling ratio
            frac = (fractions.Fraction(self._fs)
                / fractions.Fraction(self._fsOut))
            self._nIn  = frac.numerator
            self._nOut = frac.denominator

            if self._nIn > self._fs*5:
                raise ValueError('Numerator of `fs`/`fsOut` is too large')

            self._buffer = misc.CircularBuffer(
                (self._channels, int(self._fs*10)))

    def write(self, data, source=None):
        data = self._verifyData(data)

        if self._fs != self._fsOut:
            # buffer the input data
            self._buffer.write(data)
            ns = self._buffer.nsWritten
            ns = ns // self._nIn * self._nIn
            if ns <= self._buffer.nsRead: return
            data = self._buffer.read(to=ns)

            nIn  = data.shape[1]
            nOut = int(nIn / self._nIn * self._nOut)

            if self._last is None:
                last  = data[:,0]
                data  = data[:,1:]
            else:
                last  = self._last
                nOut += 1

            # init output data array
            dataOut = np.zeros((self._channels, nOut))
            dataOut[:, 0] = last
            dataOut[:,-1] = data[:,-1]

            # bin data points
            nBins    = nOut - 2
            dataBins = np.array_split(data[:,:-1], nBins, axis=1)

            # iterate over bins
            for i in range(nBins):
                if i < nBins-1:
                    nextBin = dataBins[i+1]
                else:
                    nextBin = data[:,-1:]

                a = dataOut[:,i:i+1]
                b = dataBins[i]
                c = np.mean(nextBin, axis=1)[:,np.newaxis]

                # find the point in current bin that makes the largest triangle
                areas  = abs(a + c - 2*b)/2
                argmax = np.argmax(areas, axis=1)

                dataOut[:,i+1] = b[np.arange(b.shape[0]), argmax]

            if self._last is not None:
                dataOut = dataOut[:,1:]

            self._last = dataOut[:,-1]

        else:
            # transparently pass the data down the pipeline
            dataOut = data

        super().write(dataOut, source)


class GrandAverage(Sampled):
    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        if self._channels is None:
            raise RuntimeError('Cannot set `mask` before number of `channels`')

        # verify type and size
        if not isinstance(mask, tuple):
            raise TypeError('`mask` should be a tuple')
        if len(mask) != self._channels:
            raise ValueError('Size of `mask` should match number of `channels`')
        for m in mask:
            if not isinstance(m, bool):
                raise ValueError('All elements must be `bool` instances')

        self._mask = mask

    def __init__(self, **kwargs):
        self._mask = None

        super().__init__(**kwargs)

    def _configured(self, params):
        super()._configured(params)

        self._mask = (True,) * self._channels

    def write(self, data, source=None):
        data = self._verifyData(data)

        data2 = data.copy().astype(np.float64)
        for channel in range(self._channels):
            mask = np.array(self._mask)
            if not mask[channel]: continue
            mask[channel] = False
            if not mask.any(): continue
            data2[channel,:] -= data[mask,:].mean(axis=0)

        super().write(data2, source)


class CircularBuffer(Sampled):
    @property
    def data(self):
        '''The underlying buffer.'''
        return self._data

    @property
    def size(self):
        '''Size of the underlying buffer along the circular axis.'''
        return self._size

    @property
    def shape(self):
        '''Shape of the underlying buffer: channels x circular axis size.'''
        return self._data.shape

    @property
    def ns(self):
        '''Number of samples written.'''
        return self._ns

    def __init__(self, size, **kwargs):
        if size < 1:            raise ValueError('`size` should be >= 1')
        if round(size) != size: raise TypeError ('`size` should be an integer')

        self._size = size
        self._data = None
        self._ns   = 0

        super().__init__(**kwargs)

    def _configured(self, params):
        super()._configured(params)

        self._data = np.zeros((self._channels, self._size))

    def write(self, data, source=None):
        data = self._verifyData(data)

        n    = data.shape[1]
        inds = np.arange(self._ns, self._ns + n) % self._size

        self._data[:,inds] = data
        self._ns += n

        super().write(data, source)

    def read(self, n=None):
        '''Read the last `n` samples written in the buffer.'''

        if n is None:      n = self._size
        if n < 0:          raise ValueError('`n` should be >= 0')
        if self._size < n: raise ValueError('`n` should be <= `size`')
        if self._ns < n:   raise ValueError('`n` should be <= `ns`')
        if round(n) != n:  raise TypeError ('`n` should be an integer')

        inds = np.arange(self._ns - n, self._ns) % self._size

        return self._data[:,inds]


if __name__ == '__main__':
    head   = Sampled(fs=10, channels=2)
    thread = Thread()
    split  = Split()
    filt   = LFilter(fh=4)
    ds     = DownsampleAverage(ds=2)
    avg    = GrandAverage()
    mult2  = Func(func=lambda x: x*2)
    plus1  = Func(func=lambda x: x+1)
    plus2  = Func(func=lambda x: x+2)
    pr     = Print()
    buffer = CircularBuffer(10)

    head | thread | split | (mult2,  plus1 | plus2 | filt) | pr
    # head | split | DummySink(2)
    # head | thread | avg | pr | buffer
    # head | split | (mult2, plus1) | pr

    # avg.mask = (True, True, False)

    head.write(np.array([[3, 1, 4, 6, 7, .1, 4], [12, 22, 43, 18, 32, 12, 23]]))
    head.wait()

    # print(split._params)
    # print(mult2._params)
    # print(plus1._params)

    # filt.fh = None
    # head.write(np.array([10,10,10,10,10,10,9,8,7,6,5,4]))
    # filt.fh = 9
    # head.write(np.array([10,10,10,10,10,10,9,8,7,6,5,4]))

    # (func1, func2) | pr
    #
    # func1.write(np.array([1,2,3,4,5,6,7,8,9,4]))
    # func2.write(np.array([1,2,3,4,5,6,7,8,9,4]))

    # print()
    # node = pr
    # print('Inputs: ', node._inputs )
    # print('Outputs:', node._outputs)
    # print('Sources:', node._sources)
    # print('Sinks:  ', node._sinks  )
