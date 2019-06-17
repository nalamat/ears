'''User interface for calibration of sound stimulus.


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
import logging
import scipy.io.wavfile
import numpy            as     np
import scipy            as     sp
from   PyQt5            import QtCore, QtWidgets, QtGui

import gui
import daqs
import misc
import config
import context
import platform
import globals          as     gb


log = logging.getLogger(__name__)


class CalibrationWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.initLocals()
        self.initGlobals()
        self.initDAQ()

        self.addToolBar      (QtCore.Qt.TopToolBarArea, self.initControlBar())
        self.setCentralWidget(self.initForm())
        self.setWindowTitle  (config.APP_NAME + ' Calibration')
        self.setWindowIcon   (QtGui.QIcon(config.APP_LOGO))

        self.updateSound()

    def initLocals(self):
        self.active  = False
        self.file    = ''
        self.data    = None
        self.amp     = -50
        self.startNS = 0

    def initGlobals(self):
        pass

    def initDAQ(self):
        daqs.init()
        daqs.analogOutput.dataNeeded.connect(self.analogOutputDataNeeded)

    def initControlBar(self):
        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(1)
        layout.addStretch()

        self.buttons = misc.Dict()
        buttonList = [
            # 'add', 'remove', '|',
            'load', 'save', '|',
            'start', 'pause', '|', 'close']

        for name in buttonList:
            if name == '|':
                # add vertical separator line
                layout.addWidget(gui.QVSeparator())
            else:
                # generate and add button
                btn = QtWidgets.QToolButton()
                btn.setText(name.title())
                iconFile = config.ICONS_DIR + '%s.svg' % name
                if os.path.isfile(iconFile): btn.setIcon(QtGui.QIcon(iconFile))
                btn.setIconSize((QtCore.QSize(40,32)))
                btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
                btn.setAutoRaise(True)
                # TODO: why are buttons 25% wider in mac?
                if platform.system()=='Darwin':
                    btn.setFixedWidth(btn.sizeHint().width()*.8)
                if hasattr(self, '%sClicked' % name):
                    btn.clicked.connect(getattr(self, '%sClicked' % name))
                layout.addWidget(btn)
                self.buttons[name] = btn
        layout.addStretch()

        self.buttons.pause.setEnabled(False)

        frame = QtWidgets.QFrame()
        frame.setLayout(layout)

        bar = QtWidgets.QToolBar('Control')
        bar.setMovable(False)
        bar.setStyleSheet('.QToolBar{border-bottom:%s}' % config.BORDER_STYLE)
        bar.addWidget(frame)

        return bar

    def initForm(self):
        layout = QtWidgets.QFormLayout()
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self.lstFiles = QtWidgets.QListWidget()
        for file in os.listdir(config.STIM_DIR):
            if os.path.splitext(file)[1] == config.STIM_EXT:
                self.lstFiles.addItem(file)
        self.lstFiles.setCurrentRow(0)
        self.lstFiles.currentItemChanged.connect(self.fileChanged)
        layout.addRow('Sound files:', self.lstFiles)

        self.spnAmp = QtWidgets.QSpinBox()
        self.spnAmp.setMinimum(-100)
        self.spnAmp.setMaximum(0)
        self.spnAmp.setValue(self.amp)
        self.spnAmp.valueChanged.connect(self.ampChanged)
        layout.addRow('Amplification (dB):', self.spnAmp)

        self.txtCal = QtWidgets.QLineEdit()
        self.txtCal.textEdited.connect(self.calChanged)
        layout.addRow('Calibration (dB SPL):', self.txtCal)

        frame = QtWidgets.QFrame()
        frame.setLayout(layout)

        return frame

    @gui.showExceptions
    def closeEvent(self, event):
        daqs.stop()
        daqs.clear()

    @gui.showExceptions
    def loadClicked(self, *args):
        file     = gb.session.calibrationFile.value
        dataDir  = gb.session.dataDir.value
        if file:
            file = misc.absolutePath(file, dataDir)
        else:
            file = dataDir
        file     = gui.openFile(config.SETTINGS_FILTER, file)

        if not file:
            return

        # use a temp copy for loading new calibration
        calibrationTemp = gb.calibration.copy()
        log.info('Loading calibration from "%s"', file)
        calibrationTemp.loadFile(file, float)

        # on successful load, copy the temp back to the original
        calibrationTemp.copyTo(gb.calibration)

        file = misc.relativePath(file, dataDir)
        gb.session.calibrationFile.value = file
        gb.session.saveFile(config.LAST_SESSION_FILE)

        self.updateCal()

    @gui.showExceptions
    def saveClicked(self, *args):
        file     = gb.session.calibrationFile.value
        dataDir  = gb.session.dataDir.value
        if file:
            file = misc.absolutePath(file, dataDir)
        else:
            file = dataDir
        file     = gui.saveFile(config.SETTINGS_FILTER, file)

        if not file:
            return

        log.info('Saving calibration to "%s"', file)
        gb.calibration.saveFile(file)

        file = misc.relativePath(file, dataDir)
        gb.session.calibrationFile.value = file
        gb.session.saveFile(config.LAST_SESSION_FILE)

    @gui.showExceptions
    def startClicked(self, *args):
        if not self.active:
            self.active = True
            self.updateSound()
            daqs.start()
            self.buttons.start.setEnabled(False)
            self.buttons.pause.setEnabled(True)

    @gui.showExceptions
    def pauseClicked(self, *args):
        if self.active:
            self.active = False
            daqs.stop()
            self.buttons.start.setEnabled(True)
            self.buttons.pause.setEnabled(False)

    @gui.showExceptions
    def closeClicked(self, *args):
        self.close()

    @gui.showExceptions
    def fileChanged(self, *args):
        self.updateSound()
        self.updateCal()

    @gui.showExceptions
    def ampChanged(self, *args):
        self.updateSound()
        self.updateCal()

    @gui.showExceptions
    def calChanged(self, *args):
        try:
            value = float(self.txtCal.text())-self.spnAmp.value()
        except:
            value = None

        file = self.lstFiles.currentItem().text()
        if file in gb.calibration:
            if value is None:
                gb.calibration.remove(gb.calibration[file])
            else:
                gb.calibration[file].value = value
        elif value is not None:
                gb.calibration.append(
                    context.Item(name=file, type=float, value=value))

        log.info('Saving %g', value)

    def analogOutputDataNeeded(self, task, nsWritten, nsNeeded):
        log.debug('%d samples needed at %d', nsNeeded, nsWritten)
        # prepare sound
        data = self.getSound(nsWritten, nsNeeded)
        return data

    def updateCal(self):
        file = self.lstFiles.currentItem().text()
        if file in gb.calibration:
            self.txtCal.setText('%g' %
                (gb.calibration[file].value+self.spnAmp.value()) )
        else:
            self.txtCal.setText('')

    def getSound(self, ns, length):
        '''Get a cicular window into the specified sound data.

        Args:
            ns (int): Number of sample to start the window from. This number
                will be substracted by `self.startNS`.
            length (int): The length of the circular window.
        '''
        if not self.file or self.data is None or len(self.data)==0:
            # log.info('empty data %s, %s, %s', self.file, self.data)
            return np.zeros(length)

        scale   = 10**((self.amp)/20)
        window  = np.arange(ns-self.startNS, ns-self.startNS+length)
        window %= len(self.data)
        return self.data[window] * scale

    def getRamp(self, length):
        '''Get a sin^2 ramp with the specified length.'''
        return np.sin(np.pi/2*np.arange(length)/length)**2

    def updateSound(self):
        # do nothing if neither sound file nor sound amplification have changed
        newFile = self.lstFiles.currentItem().text()
        newAmp  = self.spnAmp.value()
        if self.file == newFile and self.amp == newAmp:
            return

        # load new sound file
        if self.file != newFile:
            try:
                try:
                    log.info('Loading sound file: "%s"', newFile)
                    file = misc.absolutePath(newFile, config.STIM_DIR)
                    fs, data = sp.io.wavfile.read(file, mmap=True)
                except Exception as e:
                    raise ValueError('Cannot load sound file: "%s"' %
                        newFile) from e
                # check for sampling frequency
                if fs != daqs.analogOutput.fs:
                    raise ValueError('Sampling frequency of sound file: "%s" '
                        'should be %g' % (newFile, daqs.analogOutput.fs))
                # check if file is empty
                if not len(data):
                    raise ValueError('Specified sound file: "%s" is empty' %
                        newFile)
                data  = data.astype('float64')
                data -= data.mean()
                data *= 1/np.sqrt((data**2).mean())

                amp1 = np.log10(daqs.MIN_VOLTAGE/data.min())*20
                amp2 = np.log10(daqs.MAX_VOLTAGE/data.max())*20
                self.spnAmp.blockSignals(True)
                self.spnAmp.setMaximum(np.floor(max(amp1,amp2)))
                self.spnAmp.blockSignals(False)
                newAmp = self.spnAmp.value()
            except:
                log.exception('')
                gui.showException()
                data = None

        fs = daqs.analogOutput.fs
        # where to insert the updated sound
        ns = daqs.analogOutput.nsGenerated
        if self.active:
            ns += int(daqs.UPDATE_DELAY * fs)

        dataLength     = int(daqs.analogOutput.dataChunk * fs)
        rampLength     = int(.5 * fs)
        ramp           = self.getRamp(rampLength)

        # ramp down old masker and zero-pad
        oldData  = self.getSound(ns, rampLength)
        oldData *= ramp[::-1]
        oldData  = np.r_[oldData, np.zeros(dataLength-rampLength)]

        if self.file != newFile:
            self.data = data
            # save starting sample of the new sound
            self.startNS = ns
        # update local copies of masker file and level
        self.file = newFile
        self.amp  = newAmp

        # ramp up new masker
        newData                = self.getSound(ns, dataLength)
        newData[0:rampLength] *= ramp

        # mix old and new sound data
        data = oldData + newData

        log.debug('Updating analog output buffer at %.3f for %.3f duration '
            '(generation at %.3f)' % (ns/fs, dataLength/fs,
            daqs.analogOutput.nsGenerated/fs))
        daqs.analogOutput.write(data, ns)
