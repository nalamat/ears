'''Thread-safe storage of data in HDF5 file format using pytables package.


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

import logging
import threading
import tables    as tb


log   = logging.getLogger(__name__)

_fh    = None
_lock  = threading.Lock()
_nodes = {}

def open(*args, **kwargs):
    global _fh
    with _lock:
        if _fh is not None:
            raise IOError('Another HDF5 file is open')
        log.info('Opening HDF5 file "%s"' % args[0])
        _fh = tb.open_file(*args, **kwargs)

def close():
    global _fh
    with _lock:
        _nodes.clear()
        if _fh is not None:
            log.info('Closing HDF5 file')
            _fh.close()
            _fh = None

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

def clearTable(node):
    with _lock:
        if node not in _nodes:
            raise ValueError('Specified node %s doesn\'t exist' % node)

        _nodes[node].flush()
        _nodes[node].remove_rows(0, _nodes[node].nrows)

def createEArray(node, *args, **kwargs):
    with _lock:
        _checkFile()
        path, name = _parseNode(node)
        log.debug('Creating earray %s', node)
        _nodes[node] = _fh.create_earray(path, name, *args,
            createparents=True, **kwargs)

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


if __name__ == '__main__':
    x = None

    def test():
        global x
        x = 12

    print(x)
    test()
    print(x)
