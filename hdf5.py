'''Thread-safe storage of data in HDF5 file format using pytables package.


This file is part of the EARS project: https://github.com/nalamat/ears
Copyright (C) 2017-2019 Nima Alamatsaz <nima.alamatsaz@gmail.com>
Copyright (C) 2017-2019 NESH Lab <ears.software@gmail.com>

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

import queue
import logging
import functools
import threading
# import multiprocessing
import tables          as tb

import misc


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
