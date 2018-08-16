'''Pipeline model for routing and online processing of data streams.


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
import queue
import logging
import threading
import scipy.signal
import numpy        as np
import scipy        as sp

import misc


log = logging.getLogger(__name__)


class Route():
    '''Store input and output nodes of a pipeline route.

    Routes can be connected to other Route instance(s) using the | operator.

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
        if not routes:
            raise ValueError('`routes` cannot be empty')

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

    def _addSinks(self, sinks):
        raise NotImplementedError()

    def _addSources(self, sources):
        raise NotImplementedError()


class Node(Route):
    '''Receive data, allow manipulation and pass it downstream.

    Node
    '''

    # @property
    # def fs(self):
    #     return self._fs

    # @property
    # def sinks(self):
    #     return self._sinks

    def __init__(self, **kwargs):
        '''Keyword arguments are passed down as `params` to sinks. Depending on
        child class implementation, downstream nodes may choose to use them,
        stay indifferent, or modify them before passing on to their sinks.
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
        if not params or params == self._params:
            return

        if self._params:
            raise RuntimeError('Node can be configured only once')

        if sinkParams is None:
            sinkParams = params

        self._params     = params
        self._sinkParams = sinkParams

        # pass sinkParams to downstream nodes
        for sink in self._sinks:
            sink._config(sinkParams)

        self._configured()

    def _configured(self):
        pass

    def write(self, data, source=None):
        '''Write a chunk of data to the node for processing.

        Processed data are passed onto sink nodes.
        '''

        # transparently write data to all sinks
        for sink in self._sinks:
            sink.write(data, self)

    def wait(self):
        '''Wait for asynchronous sinks in the pipeline to finish processing.'''

        for sink in self._sinks:
            sink.wait()


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
        while not self._queue.empty():
            time.sleep(50e-3)

    def write(self, data, source=None):
        if not self._thread.isAlive():
            self._thread.start()
        self._queue.put((data, source))


class Sampled(Node):
    ''''''

    def __init__(self, **kwargs):
        self._fs       = None
        self._channels = None

        super().__init__(**kwargs)

    # def _config(self, params, sinkParams=None):
    #     if not params or params == self._params:
    #         return
    #
    #     super()._config(params, sinkParams)

    def _configured(self):
        super()._configured()

        if 'fs' not in self._params or 'channels' not in self._params:
            raise ValueError('Sampled node requires `fs` and `channels`')

        self._fs       = self._params['fs']
        self._channels = self._params['channels']

    def _verifyData(self, data):
        if data.ndim == 1: data = data[np.newaxis, :]
        if data.ndim != 2: raise ValueError('`data` should be 1D or 2D')

        if self._fs is None or self._channels is None:
            raise RuntimeError('Node is not configured')

        if self._channels != data.shape[0]:
            raise ValueError('`data` channel count does not match `channels`')

        return data


class Split(Sampled):
    '''Split multichannel data into multiple sink nodes.'''

    def _config(self, params, sinkParams=None):
        if not params or params == self._params:
            return False

        if sinkParams is None:
            sinkParams = params.copy()

        if 'channels' in sinkParams:
            # each sink will only receive 1 channel
            sinkParams['channels'] = 1

        super()._config(params, sinkParams)

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

    def _configured(self):
        super()._configured()
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


class DownsampleAverage(Sampled):
    def __init__(self, ds, **kwargs):
        if ds < 1:          raise ValueError('`ds` should be >= 1')
        if round(ds) != ds: raise ValueError('`ds` should be an integer')

        self._ds     = ds
        self._buffer = None

        super().__init__(**kwargs)

    def _config(self, params, sinkParams=None):
        if not params or params == self._params:
            return False

        if sinkParams is None:
            sinkParams = params.copy()

        if 'fs' in sinkParams:
            sinkParams['fs'] = sinkParams['fs'] / self._ds

        super()._config(params, sinkParams)

    def _configured(self):
        super()._configured()
        self._buffer = misc.CircularBuffer((self._channels, int(self._fs*10)))

    def write(self, data, source=None):
        data = self._verifyData(data)

        if self._ds != 1:
            # TODO: find a better solution than writing to a buffer
            # all nodes should be able to process very large signals
            self._buffer.write(data)
            ns = self._buffer.nsWritten
            ns = ns // self._ds * self._ds
            if ns <= self._buffer.nsRead: return
            data = self._buffer.read(to=ns)

            data = data.reshape(data.shape[0], -1, self._ds)
            data = data.mean(axis=2)

        super().write(data, source)


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

    def _configured(self):
        super()._configured()
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

    def _configured(self):
        super()._configured()
        self._data = np.zeros((self._channels, self._size))

    def write(self, data, source=None):
        data = self._verifyData(data)

        n    = data.shape[1]
        inds = np.arange(self._ns, self._ns + n) % self._size

        self._data[:,inds] = data
        self._ns += n

        super().write(data, source)

    def read(self, n):
        '''Read the last `n` samples written in the buffer.'''

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
    filt   = LFilter(fh=9)
    ds     = DownsampleAverage(ds=2)
    avg    = GrandAverage()
    mult2  = Func(func=lambda x: x*2)
    plus1  = Func(func=lambda x: x+1)
    plus2  = Func(func=lambda x: x+2)
    pr     = Print()
    buffer = CircularBuffer(10)

    head | thread | split  | (mult2,  plus1 | plus2 | filt) | pr
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
