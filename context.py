'''Encapsualtion of data and their associated widgets.


This file is part of the EARS project <https://github.com/nalamat/ears>
Copyright (C) 2017-2021 Nima Alamatsaz <nima.alamatsaz@gmail.com>
'''

import copy
import logging
import numpy     as np
import pandas    as pd
from   PyQt5     import QtCore, QtWidgets, QtGui

import hdf5
import config


log = logging.getLogger(__name__)


class Item():
    '''Bundle parameters with their UI component.'''

    @property
    def shortLabel(self):
        return self.label.split('(')[0].strip()

    @property
    def keys(self):
        if self.type != dict:
            raise NotImplementedError('`keys` only defined for Item(type=dict)')

        if self.value:
            return list(self.value.keys())
        else:
            return []

    def __init__(self, name='', label='', type=None, value=None, values=None,
            link=False, widget=None, widget2=None):
        self.name    = name
        self.label   = label
        self.type    = type
        self.value   = value
        self.values  = values
        self.link    = link
        self.widget  = widget
        self.widget2 = widget2

    def __repr__(self):
        return '%s: %s' % (repr(self.name), repr(self.value))

    def __setattr__(self, name, value):
        '''Link item value with widget.'''
        if (name=='value' and hasattr(self, 'link') and self.link
                and hasattr(self, 'widget') and self.widget):
            self.setWidgetValue(value)
        super().__setattr__(name, value)

    def getWidgetValue(self):
        if self.widget is None:
            raise ValueError('Item has no widget')

        if isinstance(self.widget, QtWidgets.QCheckBox):
            return self.widget.checkState()==QtCore.Qt.Checked

        elif isinstance(self.widget, QtWidgets.QLineEdit):
            return self.widget.text()

        elif isinstance(self.widget, QtWidgets.QComboBox):
            return self.widget.currentText()

        elif isinstance(self.widget, QtWidgets.QPlainTextEdit):
            return self.widget.toPlainText()

        else:
            raise NotImplementedError('Unspported widget "%s"' %
                type(self.widget))

    def setWidgetValue(self, value):
        if self.widget is None:
            raise ValueError('Item has no widget')

        if isinstance(self.widget, QtWidgets.QCheckBox):
            check = QtCore.Qt.Checked if self.value else QtCore.Qt.Unchecked
            self.widget.setCheckState(check)

        elif isinstance(self.widget, QtWidgets.QLineEdit):
            self.widget.setText(str(value))

        elif isinstance(self.widget, QtWidgets.QComboBox):
            self.widget.setCurrentText(str(value))

        elif isinstance(self.widget, QtWidgets.QPlainTextEdit):
            self.widget.setPlainText(str(value))

        else:
            raise NotImplementedError('Unsupported widget "%s"' %
                type(self.widget))

    def copy(self):
        return self.copyTo(Item())

    def copyTo(self, other, copyLabel=True, copyWidgets=True):
        '''Deep copy item, but maintain reference to its widgets.'''
        for (attrName, attrValue) in self.__dict__.items():
            if not copyLabel and attrName == 'label':
                continue
            if not copyWidgets and attrName in ('widget', 'widget2'):
                continue
            if isinstance(attrValue, (list,dict)):
                attrValue = copy.deepcopy(attrValue)
            other.__dict__[attrName] = attrValue
        return other



class Context(list):
    '''All-in-one class for organizing and storing `Item` objects.

    Easy access to items using iteration, numerical indexing, key name indexing
    and attribute-like access with the `.` notation.

    Store item values in HDF5 file format by initializing with `hdf5Node` and
    using `appendData` and `overwriteData` functions.

    Save and load item values in JSON format with `saveFile` and `loadFile`.

    Keep a history of item values in `dataFrame` when `appendData` is called.
    '''

    @property
    def dataFrame(self):
        return self._dataFrame

    def __init__(self, *args):
        for arg in args:
            if not isinstance(arg, Item):
                raise TypeError('Context only accepts instances of Item')
        super().__init__([*args])

        self._dataFrame = None
        self._hdf5Node  = None

    def __contains__(self, item):
        if isinstance(item, str):
            for item2 in self:
                if item2.name == item:
                    return True
            return False
        else:
            return super().__contains__(item)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.__getattr__(key)
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            return self.__setattr__(key)
        else:
            return super().__setitem__(key, value)

    def __getattr__(self, name):
        for item in self:
            if item.name == name:
                return item
        raise NameError('Name not found')

    def __setattr__(self, name, value):
        if name in self:
            raise ValueError('Cannot set value')
        super().__setattr__(name, value)

    def __repr__(self):
        # replace brackets with curly braces
        return '{%s}' % super().__repr__()[1:-1]

    def copy(self, *args):
        '''Create a new Context, deep copy all items to and return it.'''
        return Context(*[item.copy() for item in self], *args)

    def copyTo(self, other, copyTypeless=True,
            copyLabel=True, copyWidgets=True):
        '''Deep copy items to another context, replacing existing ones.'''
        for item in self:
            if not copyTypeless and item.type is None:
                continue

            if item.name in other:
                item.copyTo(other[item.name], copyLabel, copyWidgets)
            else:
                itemCopy = item.copy()
                other.append(itemCopy)
                if not copyLabel:
                    itemCopy.label = ''
                if not copyWidgets:
                    itemCopy.widget  = None
                    itemCopy.widget2 = None

    def clearValues(self):
        for item in self:
            if item.type:
                item.value = item.type()

    def applyWidgetValues(self):
        '''Transfer widget values to the context.'''
        for item in self:
            if item.widget:
                item.value = item.getWidgetValue()

    def revertWidgetValues(self):
        '''Transfer context values to widgets.'''
        for item in self:
            if item.widget:
                item.setWidgetValue(item.value)

    def initData(self, hdf5Node=None, asString=False, columnHeaders=True):
        '''Initialize data storage in HDF5 file and `dataFrame`.

        Args:
            hdf5Node (str): Path of the node to store data in HDF5 file.
                Defaults to None.
            asString (str): Store all Item values in HDF5 file as strings
                instead of their associated type (used in `globals.paradigm`
                context). Defaults to False.
            columnHeaders (bool): When True, Item names will allocate column
                headers, and their associated values are stored below them.
                If False, there will be only two columns: `name` and `value`,
                and Items are stored as rows (used in `gloabals.calibration`).
                Defaults to True.
        '''
        if columnHeaders:
            names = []
            for item in self:
                if item.type is None: continue
                names += [item.name]
            self._dataFrame = pd.DataFrame(columns=names)
        else:
            self._dataFrame = pd.DataFrame(columns=['name','value'])

        self._hdf5Node = hdf5Node

        if hdf5Node is not None:
            if hdf5.contains(hdf5Node):
                raise NameError('HDF5 node %s already exists' % hdf5Node)

            if columnHeaders:
                desc    = []
                typeMap = {
                    bool : 'S5'  , int  : 'i4'  , float: 'f8'  ,
                    str  : 'S512', list : 'S512', dict : 'S512',
                    }
                for item in self:
                    # skip items with no types
                    if item.type is None: continue
                    if asString: dtype = 'S512'
                    else       : dtype = typeMap[item.type]
                    desc += [(item.name, dtype)]
            else:
                desc = [('name','S512'), ('value','S512')]

            desc = np.dtype(desc)
            hdf5.createTable(hdf5Node, desc)

        self._asString      = asString
        self._columnHeaders = columnHeaders

    def clearData(self):
        if self._dataFrame is None:
            raise ValueError('Data not initialized')

        # clear dataframe
        self._dataFrame = self._dataFrame.iloc[0:0]

        # clear HDF5 table
        if self._hdf5Node is not None:
            hdf5.clearTable(self._hdf5Node)

    def appendData(self):
        if self._dataFrame is None:
            raise ValueError('Data not initialized')

        # keep a history of item values
        if self._columnHeaders:
            data = {}
            for item in self:
                if item.type is None: continue
                data[item.name] = item.value
        else:
            data = {'name':[], 'value':[]}
            for item in self:
                data['name' ] += [item.name ]
                data['value'] += [item.value]
        self._dataFrame = self._dataFrame.append(data, ignore_index=True)

        # dump item values to HDF5 file
        if self._hdf5Node is not None:
            if self._columnHeaders:
                data = {}

                for item in self:
                    # skip items with no types or values
                    if item.type is None or item.value is None: continue
                    value = item.value
                    if self._asString or item.type in (bool, list, dict):
                        value = str(value)
                    data[item.name] = value

                hdf5.appendTable(self._hdf5Node, data)

            else:
                for item in self:
                    data = {'name':item.name, 'value':str(item.value)}
                    hdf5.appendTable(self._hdf5Node, data)

    def overwriteData(self):
        self.clearData()
        self.appendData()

    def saveFile(self, file):
        with open(file, 'w') as fh:
            fh.write('{\n')
            for item in self:
                if item.type is None:
                    continue
                fh.write('\t')
                fh.write(repr(item))
                fh.write(',\n')
            fh.write('}\n')

    def loadFile(self, file, newItemType=None):
        with open(file, 'r') as fh:
            contents = eval(fh.read())

        for (name, value) in contents.items():
            found = False
            for item in self:
                if item.name == name:
                    item.value = value
                    found = True
                    break
            if not found:
                if newItemType:
                    self.append(Item(name, '', newItemType, value))
                else:
                    log.warning('File item "%s" not found in Context', name)
