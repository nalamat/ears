'''Some utility stuff!


This file is part of the EARS project: https://github.com/nalamat/ears
Copyright (C) 2017-2020 Nima Alamatsaz <nima.alamatsaz@gmail.com>
Copyright (C) 2017-2020 NESH Lab <ears.software@gmail.com>
Distrubeted under GNU GPLv3. See LICENSE.txt for more info.
'''

import os
import queue
import logging
import pathlib
import functools
import threading
import subprocess
import collections
import numpy       as     np


log = logging.getLogger(__name__)


# def logExceptions(message=''):
#     '''Function decorator to log exceptions.'''
#
#     def wrapper(func):
#
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             try:
#                 func(*args, **kwargs)
#             except:
#                 log.exception(message)
#
#         return wrapper
#
#     return wrapper


# def inDirectory(file, directory):
#     file      = pathlib.Path(file     ).resolve()
#     directory = pathlib.Path(directory).resolve()
#
#     if file.exists() and file.samefile(directory):
#         return True
#
#     for parent in file.parents:
#         if parent.exists() and parent.samefile(directory):
#             return True
#
#     return False

def listLike(obj, strIncluded=True):
    return (hasattr(obj, '__len__') and
        (strIncluded or not isinstance(obj, str)) )

def iterable(obj):
    return isinstance(obj, collections.Iterable)

def inDirectory(file, directory):
    file      = os.path.normcase(os.path.realpath(file     ))
    directory = os.path.normcase(os.path.realpath(directory))

    return os.path.commonpath([file, directory]) == directory

def relativePath(file, directory=None):
    if not file:
        return file

    if not directory:
        directory = os.getcwd()

    if inDirectory(file, directory):
        return os.path.relpath(file, directory).replace('\\', '/')
    else:
        return file.replace('\\', '/')

def absolutePath(file, directory=None):
    if not directory:
        directory = os.getcwd()

    if os.path.isabs(file):
        return file.replace('\\', '/')
    else:
        return os.path.realpath(
            os.path.join(directory, file)).replace('\\', '/')

def getCommitHash():
    '''Return the current commit hash (HEAD)'''
    try:
        proc = subprocess.run(['git', 'log', '-1', '--pretty="%H"'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return proc.stdout.decode().strip(' \t\r\n"')
    except:
        return ''

class Queue(queue.Queue):
    def clear(self):
        while not self.empty():
            self.get()


class Dict(dict):
    '''Extended dictionary supporting attribute-like access to values using the
    `.` notation. The keys for `.` access should be legible python name strings.
    '''
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


class CircularBuffer():
    '''An efficient circular buffer using numpy array.'''

    @property
    def shape(self):
        return self._data.shape

    @property
    def axis(self):
        '''The dimension along which the buffer is circular.'''
        return self._axis

    @property
    def nsWritten(self):
        '''Total number of samples written to the buffer.'''
        return self._nsWritten

    @nsWritten.setter
    def nsWritten(self, value):
        '''Update both written and read number of samples.

        Only rewind is allowed.'''

        # if value < 0:
        #     value = self._nsWritten - value
        if value < 0:
            raise IndexError('Cannot rewind past 0')
        if self._nsWritten < value:
            raise IndexError('Cannot fast forward')

        self._nsWritten = value
        if value < self._nsRead:
            self._nsRead = value
        self._updatedEvent.set()

    @property
    def nsRead(self):
        '''Total number of samples read from the buffer.'''
        return self._nsRead

    @nsRead.setter
    def nsRead(self, value):
        '''Update the read number of samples.

        Only rewind is allowed.'''

        # if value < 0:
        #     value = self._nsRead - value
        if value < 0:
            raise IndexError('Cannot rewind past 0')
        if self._nsRead < value:
            raise IndexError('Cannot fast forward')
        if value < self._nsWritten - self._data.shape[self._axis]:
            raise IndexError('Cannot rewind past the (circular) buffer size')

        self._nsRead = value
        self._updatedEvent.set()

    @property
    def nsAvailable(self):
        '''Number of samples available but not read yet.'''
        return self._nsWritten - self._nsRead

    def __init__(self, shape, axis=-1, dtype=np.float64):
        '''
        Args:
            shape (int or tuple of int): Size of each dimension.
            axis (int): The dimension along which the buffer is circular.
            dtype (type): Data type to define the numpy array with.
        '''
        self._data         = np.zeros(shape, dtype)
        self._axis         = axis
        self._nsWritten    = 0
        self._nsRead       = 0
        self._updatedEvent = threading.Event()
        self._lock         = threading.Lock()

    def __str__(self):
        return ' nsWritten: %d\nData:\n%s' % (self._nsWritten, self._data)

    # def __len__(self):
    #     '''Total number of samples written to buffer.'''
    #     return self._nsWritten

    def __enter__(self):
        self._lock.acquire()

    def __exit__(self, *args):
        self._lock.release()

    def _getWindow(self, indices):
        '''Get a multi-dimensional and circular window into the buffer.

        Args:
            indices (array-like): 1-dimensional array with absolute indices
                (i.e. relative to the first sample) along the circular axis of
                the buffer.
        '''
        indices %= self._data.shape[self._axis]
        window = [slice(None)]*self._data.ndim
        window[self._axis] = indices
        return tuple(window)

    def write(self, data, at=None):
        '''Write samples to the end of buffer.

        Args:
            data (array-like): All dimensions of the given `data` should match
                the buffer's shape except along the cicular `axis`.
            at (int): ...
        '''

        # convert to numpy array
        if isinstance(data, list): data = np.array(data)

        if at is None:
            at = self._nsWritten
        if at < 0:
            at = self._nsWritten - at
        if at < 0:
            raise IndexError('Cannot write before 0')
        if self._nsWritten < at:
            raise IndexError('Cannot skip and write (write: %d, at: %d)' %
                (self._nsWritten, at))
        if at < self._nsRead:
            raise IndexError('Cannot write before last read sample ('
                'at: %d, nsWrite: %d, nsRead: %d)' %
                (at, self._nsWritten, self._nsRead) )

        # prepare circular write indices
        indices = np.arange(at, at + data.shape[self._axis])
        window = self._getWindow(indices)
        # write data to buffer
        self._data[window] = data
        # update written number of sample
        self._nsWritten = at + data.shape[self._axis]
        # check for buffer overflow
        if self._nsRead < self._nsWritten - self._data.shape[self._axis]:
            raise BufferError('Circular buffer overflow occured (%d, %d, %d)' %
                (self._nsRead,self._nsWritten,self._data.shape[self._axis]))
        self._updatedEvent.set()

    def read(self, frm=None, to=None, advance=True):
        '''Read samples from the buffer.

        Read data might have to be copied for thread safety and in order to
        prevent being overwritten before being processed.

        Args:
            frm (int): Start index for reading data. Defaults to None which
                reads from the last read sample.
            to (int): End index for reading data. Negative values indicate an
                end index relative to 'nsWritten' (last sample written).
                Defaults to None which reads up to last available sample.
        '''

        # get value here to avoid racing condition
        nsWritten = self._nsWritten
        if frm is None: frm = self._nsRead
        if to  is None: to  = nsWritten
        if to < 0: to = nsWritten - to

        if to < frm:
            raise IndexError('Cannot read less negative number of samples')
        if frm < self._nsWritten - self._data.shape[self._axis]:
            raise IndexError('Cannot read past (circular) buffer size')
        if nsWritten < to:
            raise IndexError('Cannot read past last written sample')

        indices = np.arange(frm, to)

        # without any locks there is still a chance for racing condition leading
        # to buffer overflow when new data is written after the boundary check
        # and before the returned data (by reference) is used
        window = self._getWindow(indices)

        # advance number of samples written
        if advance:
            self._nsRead = to

        # data should be copied after returning for thread safety
        return self._data[window]

    def wait(self):
        self._updatedEvent.wait()
        self._updatedEvent.clear()

    def updated(self):
        result = self._updatedEvent.isSet()
        if result: self._updatedEvent.clear()
        return result


class Event():
    '''An event that can handle multiple callback functions.'''

    def __init__(self, callback=None, singleCallback=False):
        '''
        Args:
            callback (callable): The function or callable object to initialize
                the event with. On invocation of the event, this object will
                be called. Defaults to None.
            singleCallback (bool): Whether the event only accepts a single
                callback. If True, connecting a new callback will disconnect
                the previous one. Also on event invocation, function return
                values will only be returned in single callback mode. Defaults
                to False.
        '''
        self._callbacks = []
        self._singleCallback = singleCallback

        if callback is None:
            pass
        elif callable(callback):
            self._callbacks += [callback]
        else:
            raise ValueError('The `callback` object must be callable')

    def connect(self, callback):
        '''Connect a function or callable object to the event.'''
        if not callable(callback):
            raise ValueError('The `callback` object must be callable')

        if self._singleCallback:
            self._callbacks = [callback]
        else:
            self._callbacks += [callback]
            self._callbacks = list(set(self._callbacks))    # unique!

    def disconnect(self, callback):
        '''Remove a function or callable object from event.'''
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def clear(self):
        self._callbacks.clear()

    def __len__(self):
        return len(self._callbacks)

    def __call__(self, *args, **kwargs):
        if self._singleCallback:
            # return callback return values in single callback mode
            return self._callbacks[0](*args, **kwargs)
        else:
            # ignore callback return values, even if there is only one callback
            for callback in self._callbacks:
                callback(*args, **kwargs)


if __name__=='__main__':
    # stream = CircularBuffer((2,10))
    # print(stream)
    # stream.write(np.array([np.arange(0,5), np.arange(100,105)]))
    # print(stream)
    # stream.write(np.array([np.arange(5,15), np.arange(105,115)]))
    # print(stream)
    # print(stream.read(np.arange(7,14)))
    # print(len(stream))

    # @logExceptions()
    # def func():
    #     raise ValueError()
    #
    # func()

    print(getCommitHash())
