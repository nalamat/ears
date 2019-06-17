'''Utility functions, decorators and classes to help Qt GUI design.


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

import os
import sys
import logging
import traceback
import functools
from   PyQt5     import QtCore, QtWidgets, QtGui

import config


log = logging.getLogger(__name__)


def showError(text1, text2=None, text3=None, title='Error'):
    showNormalCursor()

    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Critical)
    msg.setText(text1)
    if text2: msg.setInformativeText(text2)
    if text3: msg.setDetailedText(text3)
    msg.setWindowTitle(title)
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
    msg.setWindowIcon(QtGui.QIcon(config.APP_LOGO))

    msg.exec_()


def showException(e=None):
    try:
        if e: raise e
    finally:
        text1 = str(sys.exc_info()[1])
        if not text1: text1 = 'Unhandled exception occured'
        text2 = traceback.format_exc(-1)
        text3 = traceback.format_exc()
        showError(text1, text2, text3, 'Exception')


def showExceptions(func):
    '''Function decorator to log and show exceptions during GUI callbacks.'''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            log.exception('')
            showException()

    return wrapper


def logExceptions(func):
    '''Function decorator to log exceptions.'''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except:
            log.exception('')

    return wrapper


def showQuestion(text1, text2=None, title=config.APP_NAME):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Question)
    msg.setText(text1)
    if text2: msg.setInformativeText(text2)
    msg.setWindowTitle(title)
    msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
    msg.setWindowIcon(QtGui.QIcon(config.APP_LOGO))

    if msg.exec_() == QtWidgets.QMessageBox.Yes:
        return True
    else:
        return False


def getFile(acceptMode, fileMode, nameFilters='', select=''):
    if isinstance(nameFilters, str):
        nameFilters = [nameFilters]

    dlg = QtWidgets.QFileDialog()
    dlg.setAcceptMode(acceptMode)
    dlg.setFileMode(fileMode)
    dlg.setNameFilters(nameFilters)

    if select:
        select = os.path.realpath(select)
        if os.path.isdir(select) or fileMode in (
                QtWidgets.QFileDialog.Directory,
                QtWidgets.QFileDialog.DirectoryOnly):
            dlg.setDirectory(select)
        else:
            directory, file = os.path.split(select)
            dlg.setDirectory(directory)
            dlg.selectFile(file)

    # dlg.setWindowTitle('Test')

    if dlg.exec_() == QtWidgets.QDialog.Accepted:
        return dlg.selectedFiles()[0]
    else:
        return None


def openFile(nameFilters='All files (*)', select=''):
    return getFile(QtWidgets.QFileDialog.AcceptOpen,
        QtWidgets.QFileDialog.ExistingFile, nameFilters, select)


def saveFile(nameFilters='All files (*)', select=''):
    return getFile(QtWidgets.QFileDialog.AcceptSave,
        QtWidgets.QFileDialog.AnyFile, nameFilters, select)


def openDirectory(select=''):
    return getFile(QtWidgets.QFileDialog.AcceptOpen,
        QtWidgets.QFileDialog.DirectoryOnly, '', select)


def makeFileButton(toolTip=''):
    btn = QtWidgets.QPushButton()
    btn.setIcon(QtGui.QIcon(config.ICONS_DIR + 'folder.svg'))
    btn.setFlat(True)
    btn.setStyleSheet('padding:0px')
    btn.setToolTip(toolTip)
    btn.setMaximumWidth(btn.sizeHint().width())
    return btn


def centerWindow(window, screenNumber=-1):
    '''Center window in the specified screen.

    Args:
        screenNumber (int): Use -1 for the primary screen and -2 for an
            alternate screen (if available). Defaults to the primary screen.
    '''
    desktop = QtWidgets.QApplication.desktop()

    if screenNumber == -1:
        # primary screen
        screenNumber = desktop.primaryScreen()
    elif screenNumber == -2:
        # alternate screen
        screenNumber = (desktop.primaryScreen()+1) % desktop.screenCount()
    elif 0 <= screenNumber < desktop.screenCount():
        # selected screen
        pass
    else:
        # invalid screen number, use primary screen
        screenNumber = desktop.primaryScreen()

    window.move(desktop.availableGeometry(screenNumber).center() -
        window.rect().center())


def setCheckState(checkBox, checkState, blockSignals=True):
    '''Set check state of a QCheckBox while optionally blocking its signals.'''
    if not isinstance(checkBox, QtWidgets.QCheckBox):
        raise TypeError('`checkBox` should be an instance of `QCheckBox`')
    if blockSignals: checkBox.blockSignals(True)
    checkBox.setCheckState(checkState)
    if blockSignals: checkBox.blockSignals(False)


def showBusyCursor():
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)


def showNormalCursor():
    QtWidgets.QApplication.restoreOverrideCursor()


class BusyCursor():
    '''Show busy cursor when entering the context manager.

    Example:
        with BusyCursor():
            do_a_long_process()
    '''
    def __enter__(self):
        showBusyCursor()

    def __exit__(self, *args):
        showNormalCursor()


def busyCursor(func):
    '''Function decorator to show busy cursor when executing `func`.'''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with BusyCursor():
            func(*args, **kwargs)

    return wrapper


class QSeparator(QtWidgets.QFrame):
    def __init__(self, orientation, *args, **kwargs):
        super().__init__(*args, **kwargs)

        orientation = orientation.lower()
        if orientation=='v' or orientation=='ver' or orientation=='vertical':
            self.setFrameShape(QtWidgets.QFrame.VLine)
        elif orientation=='h' or orientation=='hor' or orientation=='horizontal':
            self.setFrameShape(QtWidgets.QFrame.HLine)
        else:
            raise ValueError('Invalid orientation')

        self.setFrameShadow(QtWidgets.QFrame.Raised)


class QVSeparator(QSeparator):
    def __init__(self, *args, **kwargs):
        super().__init__('v', *args, **kwargs)


class QHSeparator(QSeparator):
    def __init__(self, *args, **kwargs):
        super().__init__('h', *args, **kwargs)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    print(openFile())
