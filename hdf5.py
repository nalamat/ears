'''Thread-safe storage of data in HDF5 file format using pytables package.


This file is part of the EARS project: https://github.com/nalamat/ears
Copyright (C) 2017-2018 Nima Alamatsaz <nima.alamatsaz@njit.edu>
Copyright (C) 2017-2018 Antje Ihlefeld <antje.ihlefeld@njit.edu>
Distrubeted under GNU GPLv3. See LICENSE.txt for more info.
'''

import queue
import logging
import functools
import threading
# import multiprocessing
import tables          as tb

import pipeline


log   = logging.getLogger(__name__)

_fh      = None
_lock    = threading.Lock()
_nodes   = {}
# _queue   = misc.Queue()
# _stop    = threading.Event()
# _thread  = None

# def runInThread(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         if threading.current_thread() == _thread:
#             # log.info('################# In current thread: %s' % func)
#             return func(*args, **kwargs)
#         else:
#             # log.info('################# In another thread: %s' % func)
#             _queue.put((func, args, kwargs))
#
#     return wrapper

def open(*args, **kwargs):
    global _fh, _thread
    with _lock:
        if _fh is not None:
            raise IOError('Another HDF5 file is open')
        log.info('Opening HDF5 file "%s"' % args[0])
        _fh = tb.open_file(*args, **kwargs)
        # _stop.clear()
        # _thread = threading.Thread(target=_loop, args=(_stop,_queue))
        # _thread.daemon = True
        # log.info('Starting HDF5 thread')
        # _thread.start()

def close():
    global _fh
    # log.info('Stopping HDF5 thread')
    # _stop.set()
    # if _thread:
    #     while _thread.is_alive():
    #         _thread.join(.1)
    with _lock:
        _nodes.clear()
        if _fh is not None:
            log.info('Closing HDF5 file')
            _fh.close()
            _fh = None

# @runInThread
def flush():
    with _lock:
        _checkFile()
        _fh.flush()

def contains(node):
    with _lock:
        _checkFile()
        return _fh.__contains__(node)

def createGroup(node, *args, **kwargs):
    with _lock:
        _checkFile()
        path, name = _parseNode(node)
        log.debug('Creating group %s', node)
        _fh.create_group(path, name, *args, createparents=True, **kwargs)

def createTable(node, *args, **kwargs):
    with _lock:
        _checkFile()
        path, name = _parseNode(node)
        log.debug('Creating table %s', node)
        _nodes[node] = _fh.create_table(path, name, *args,
            createparents=True, **kwargs)

# @runInThread
def appendTable(node, data):
    '''
    Args:
        node (str): ...
        data (dict): ...
    '''
    with _lock:
        if node not in _nodes:
            raise ValueError('Specified node %s doesn\'t exist' % node)
        row = _nodes[node].row
        for key, value in data.items():
            row[key] = value
        row.append()

# @runInThread
def clearTable(node):
    with _lock:
        if node not in _nodes:
            raise ValueError('Specified node %s doesn\'t exist' % node)

        # _nodes[node].flush()
        _nodes[node].remove_rows(0, _nodes[node].nrows)

def createEArray(node, *args, **kwargs):
    with _lock:
        _checkFile()
        path, name = _parseNode(node)
        log.debug('Creating earray %s', node)
        _nodes[node] = _fh.create_earray(path, name, *args,
            createparents=True, **kwargs)

# @runInThread
def appendArray(node, data):
    '''
    Args:
        node (str): ...
        data (numpy.array): ...
    '''
    with _lock:
        if node not in _nodes:
            raise ValueError('Specified node %s doesn\'t exist' % node)
        _nodes[node].append(data)

def setNodeAttr(node, attrName, attrValue):
    with _lock:
        _checkFile()
        _fh.set_node_attr(node, attrName, attrValue)

def _parseNode(node):
    node.rstrip('/')
    path = '/'.join(node.split('/')[:-1])
    if path == '': path = '/'
    name = node.split('/')[-1]
    return (path, name)

def _checkFile():
    if _fh is None:
        raise IOError('No open HDF5 file')

# def _loop(_stop, _queue):
#     log.info('Starting HFD5 loop')
#
#     try:
#         # with nogil:
#             while not _stop.wait(.1):
#                 while not _queue.empty():
#                     log.info('Fetching from queue')
#                     (func, args, kwargs) = _queue.get()
#                     log.info('Executing function %s', func)
#                     func(*args, **kwargs)
#                     log.info('Executed function %s', func)
#     except:
#         log.exception('')
#
#     log.info('Stopping HFD5 loop')


class AnalogStorage(pipeline.Sampled):
    def __init__(self, hdf5Node, compLib='zlib', compLevel=1, **kwargs):
        '''
        Args:
            hdf5Node (str): Path of the node to store data in HDF5 file.
            compLib (str): Compression library, should be one of the following:
                zlib, lzo, bzip2, blosc, blosc:blosclz, blosc:lz4,
                blosc:lz4hc, blosc:snappy, blosc:zlib, blosc:zstd
            compLevel: Level of compression can vary from 0 (no compression)
                to 9 (maximum compression)
        '''
        if not isinstance(hdf5Node, str):
            raise TypeError('`hdf5Node` should be a string')

        self._hdf5Node    = hdf5Node
        self._hdf5Filters = tb.Filters(complib=compLib, complevel=compLevel)

        super().__init__(**kwargs)

    def _configured(self, params, sinkParams):
        super()._configured(params, sinkParams)

        if contains(self._hdf5Node):
            raise NameError('HDF5 node %s already exists' % self._hdf5Node)
        createEArray(self._hdf5Node, tb.Float32Atom(),
            (0, self._channels), '', self._hdf5Filters,
            expectedrows=self._fs*60*30)    # 30 minutes
        setNodeAttr(self._hdf5Node, 'fs', self._fs)

    def _writing(self, data, source):
        data = super()._writing(data, source)

        appendArray(self._hdf5Node, data.transpose())

        return data


if __name__ == '__main__':
    import numpy as np
    import datetime as dt

    # test bench different compression libraries for speed and size
    fs    = int(31.25e3)
    dur   = 5*60
    chunk = 1*60

    for complib in ('zlib', 'blosc', 'lzo', 'bzip2'):
        for complevel in (1,5,9):
            file = 'test_hdf5/%s-%d.h5' % (complib, complevel)
            filters = tb.Filters(complevel=complevel, complib=complib)
            node = '/trace'
            open(file, mode='w')
            createEArray(node, tb.Float32Atom(), (0,16),
                '', filters, expectedrows=dur*fs)
            setNodeAttr(node, 'fs', fs)

            tic = dt.datetime.now()
            for i in range(0, dur, chunk):
                appendArray(node, np.random.rand(chunk*fs, 16))
            flush()
            toc = (dt.datetime.now()-tic).total_seconds()
            print('%s-%d: %g' % (complib, complevel, toc))
            close()
