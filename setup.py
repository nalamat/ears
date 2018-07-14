'''Setup window allowing user to select desired experiment settings.


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

import os
import logging
import platform
import numpy      as     np
import datetime   as     dt
from   PyQt5      import QtCore, QtWidgets, QtGui

import gui
import misc
import config
import globals    as     gb


log = logging.getLogger(__name__)


class SetupWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        # unlike BehaviorWindow, graphical layout in SetupWindow requires
        # micromanaging and hence cannot be generated automatically
        dataDir = gb.session.dataDir.value
        self.dataDir = QtWidgets.QLineEdit(dataDir)
        self.dataDir.textEdited.connect(self.dataDirChanged)
        h = self.dataDir.sizeHint().height()

        self.dataDirButton = gui.makeFileButton('Select data directory')
        self.dataDirButton.clicked.connect(self.dataDirButtonClicked)

        dataDirectoryLayout = QtWidgets.QHBoxLayout()
        dataDirectoryLayout.addWidget(self.dataDir)
        dataDirectoryLayout.addWidget(self.dataDirButton)

        self.subjectID = QtWidgets.QComboBox()
        self.subjectID.setEditable(True)
        self.loadSubjectIDs()
        self.subjectID.setCurrentText(gb.session.subjectID.value)
        self.subjectID.currentTextChanged.connect(self.updateDataFile)

        self.experimentName = QtWidgets.QComboBox()
        self.experimentName.setEditable(True)
        self.experimentName.addItems(gb.session.experimentName.values)
        self.experimentName.setCurrentText(gb.session.experimentName.value)
        self.experimentName.currentTextChanged.connect(self.updateDataFile)

        self.experimentMode = QtWidgets.QComboBox()
        self.experimentMode.addItems(gb.session.experimentMode.values)
        self.experimentMode.setCurrentText(gb.session.experimentMode.value)

        self.recording = QtWidgets.QComboBox()
        self.recording.addItems(gb.session.recording.values)
        self.recording.setCurrentText(gb.session.recording.value)
        self.recording.currentIndexChanged.connect(self.updateDataFile)

        self.autoDataFile = QtWidgets.QCheckBox()
        self.autoDataFile.setToolTip('Automatic file naming')
        if gb.session.autoDataFile.value:
            self.autoDataFile.setCheckState(QtCore.Qt.Checked)
        else:
            self.autoDataFile.setCheckState(QtCore.Qt.Unchecked)
        self.autoDataFile.stateChanged.connect(self.autoDataFileChanged)

        self.dataFile = QtWidgets.QLineEdit()
        # self.dataFile.setFixedWidth(self.dataFile.sizeHint().width()*2.5)
        if gb.session.autoDataFile.value:
            self.updateDataFile()
            self.dataFile.setReadOnly(True)
        else:
            self.dataFile.setText(gb.session.dataFile.value)
            self.dataFile.setCursorPosition(0)

        self.dataFileButton = gui.makeFileButton('Select data file')
        self.dataFileButton.clicked.connect(self.dataFileButtonClicked)
        if gb.session.autoDataFile.value:
            self.dataFileButton.setEnabled(False)

        dataFileLayout = QtWidgets.QHBoxLayout()
        dataFileLayout.addWidget(self.dataFile)
        dataFileLayout.addWidget(self.autoDataFile)
        dataFileLayout.addWidget(self.dataFileButton)

        calibrationFile = gb.session.calibrationFile.value
        calibrationFile = misc.relativePath(calibrationFile, dataDir)
        self.calibrationFile = QtWidgets.QLineEdit(calibrationFile)

        self.calibrationFileButton = gui.makeFileButton(
            'Select calibration file')
        self.calibrationFileButton.clicked.connect(
            self.calibrationFileButtonClicked)

        calibrationFileLayout = QtWidgets.QHBoxLayout()
        calibrationFileLayout.addWidget(self.calibrationFile)
        calibrationFileLayout.addWidget(self.calibrationFileButton)

        paradigmFile = gb.session.paradigmFile.value
        paradigmFile = misc.relativePath(paradigmFile, dataDir)
        self.paradigmFile = QtWidgets.QComboBox()
        self.paradigmFile.setEditable(True)
        self.loadParadigmFiles()
        self.paradigmFile.setCurrentText(paradigmFile)
        self.paradigmFile.currentTextChanged.connect(self.paradigmFileChanged)

        self.paradigmFileButton = gui.makeFileButton(
            'Select paradigm file')
        self.paradigmFileButton.clicked.connect(self.paradigmFileButtonClicked)

        paradigmFileLayout = QtWidgets.QHBoxLayout()
        paradigmFileLayout.addWidget(self.paradigmFile)
        paradigmFileLayout.addWidget(self.paradigmFileButton)

        self.rove = QtWidgets.QListWidget()
        if self.paradigmFile.currentText(): self.rove.setEnabled(False)
        self.roveParams = []
        for item in gb.paradigm:
            if item.type in (int, float, str):
                self.roveParams += [item.name]
                widgetItem = QtWidgets.QListWidgetItem(item.label)
                widgetItem.setFlags(widgetItem.flags() |
                    QtCore.Qt.ItemIsUserCheckable)
                if item.name in gb.session.rove.value:
                    widgetItem.setCheckState(QtCore.Qt.Checked)
                else:
                    widgetItem.setCheckState(QtCore.Qt.Unchecked)
                self.rove.addItem(widgetItem)

        # OK and Cancel buttons
        btnOK = QtWidgets.QPushButton('OK')
        btnOK.setDefault(True)
        btnOK.clicked.connect(self.okClicked)
        # btnRevert = QtWidgets.QPushButton('Revert')
        # btnRevert.clicked.connect(self.revertClicked)
        btnCancel = QtWidgets.QPushButton('Cancel')
        btnCancel.clicked.connect(self.reject)
        btnCalibration = QtWidgets.QPushButton('Calibration')
        btnCalibration.clicked.connect(self.calibrationClicked)

        buttonsLayout = QtWidgets.QHBoxLayout()
        buttonsLayout.addStretch()
        buttonsLayout.addWidget(btnOK)
        # buttonsLayout.addWidget(btnRevert)
        buttonsLayout.addWidget(btnCancel)
        buttonsLayout.addWidget(btnCalibration)
        buttonsLayout.addStretch()

        layout = QtWidgets.QGridLayout()
        layout.addWidget(QtWidgets.QLabel('Data directory:'),
            layout.rowCount()-1, 0)
        layout.addLayout(dataDirectoryLayout, layout.rowCount()-1, 1)
        layout.addWidget(QtWidgets.QLabel('Subject ID:'), layout.rowCount(), 0)
        layout.addWidget(self.subjectID, layout.rowCount()-1, 1)
        layout.addWidget(QtWidgets.QLabel('Experiment name:'),
            layout.rowCount(), 0)
        layout.addWidget(self.experimentName, layout.rowCount()-1, 1)
        layout.addWidget(QtWidgets.QLabel('Experiment mode:'),
            layout.rowCount(), 0)
        layout.addWidget(self.experimentMode, layout.rowCount()-1, 1)
        layout.addWidget(QtWidgets.QLabel('Recording:'), layout.rowCount(), 0)
        layout.addWidget(self.recording, layout.rowCount()-1, 1)
        layout.addWidget(QtWidgets.QLabel('Data file:'), layout.rowCount(), 0)
        layout.addLayout(dataFileLayout, layout.rowCount()-1, 1)
        layout.addWidget(QtWidgets.QLabel('Calibration file:'),
            layout.rowCount(), 0)
        layout.addLayout(calibrationFileLayout, layout.rowCount()-1, 1)
        layout.addWidget(QtWidgets.QLabel('Paradigm file:'),
            layout.rowCount(), 0)
        layout.addLayout(paradigmFileLayout, layout.rowCount()-1, 1)
        layout.addWidget(QtWidgets.QLabel('Rove:'), layout.rowCount(), 0)
        layout.addWidget(self.rove, layout.rowCount()-1, 1)
        layout.addLayout(buttonsLayout, layout.rowCount(), 0, 1, 3)
        layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)

        self.setLayout(layout)
        self.setWindowTitle(config.APP_NAME + ' Setup')
        self.setWindowIcon(QtGui.QIcon(config.APP_LOGO))
        self.setWindowFlags(self.windowFlags() &
            ~QtCore.Qt.WindowContextHelpButtonHint)
        # self.raise_()
        # self.activateWindow()

    @gui.showExceptions
    def accept(self):
        dataDir = misc.absolutePath(self.dataDir.text())

        # verify data file path
        dataFile = misc.absolutePath(self.dataFile.text(), dataDir)
        if gb.appMode == 'Experiment':
            if not dataFile:
                gui.showError(
                    'Entered data file is not valid',
                    'File path cannot be empty')
                return
            dir_, file = os.path.split(dataFile)
            if dir_ and not os.path.isdir(dir_):
                res = gui.showQuestion(
                    'Directory "%s" doesn\'t exist' % dir_,
                    'Would you like to create it?')
                if not res: return
                os.makedirs(dir_)
            fileName, fileExt = os.path.splitext(file)
            if fileExt != config.DATA_EXT:
                gui.showError(
                    'Entered data file extension is not valid',
                    'Please change the extension to "%s"' % config.DATA_EXT)
                return
            if os.path.isfile(dataFile):
                gui.showError(
                    'Entered data file already exists',
                    'Please enter a different file path')
                return

        # verify calibration file path
        calibrationFile = self.calibrationFile.text()
        if calibrationFile:
            calibrationFile = misc.absolutePath(calibrationFile, dataDir)
            if not os.path.isfile(calibrationFile):
                gui.showError(
                    'Entered calibration file doesn\'t exist',
                    'Please enter a different file path')
                return

        # verify paradigm file path
        paradigmFile = self.paradigmFile.currentText()
        if paradigmFile and gb.appMode == 'Experiment':
            paradigmFile = misc.absolutePath(paradigmFile, dataDir)
            if not os.path.isfile(paradigmFile):
                gui.showError(
                    'Entered paradigm file doesn\'t exist',
                    'Please enter a different file path')
                return

        # verify rove parameters
        rove = []
        if not paradigmFile:
            for i in range(self.rove.count()):
                item = self.rove.item(i)
                if item.checkState() == QtCore.Qt.Checked:
                    rove += [self.roveParams[i]]
            if not rove and gb.appMode == 'Experiment':
                gui.showError(
                    'No rove parameters selected',
                    'Please select at least one parameter')
                return

        # transfer all widget values to gb.session
        autoDataFile = self.autoDataFile.checkState()==QtCore.Qt.Checked

        gb.session.clearValues()

        gb.session.dataDir.value            = dataDir
        gb.session.subjectID.value          = self.subjectID.currentText()
        gb.session.experimentName.value     = self.experimentName.currentText()
        gb.session.experimentMode.value     = self.experimentMode.currentText()
        gb.session.recording.value          = self.recording.currentText()
        gb.session.autoDataFile.value       = autoDataFile
        gb.session.dataFile.value           = dataFile
        gb.session.calibrationFile.value    = calibrationFile
        gb.session.paradigmFile.value       = paradigmFile
        gb.session.rove.value               = rove
        gb.session.sessionTime.value        = gb.sessionTimeComplete
        gb.session.computerName.value       = platform.node()
        gb.session.commitHash.value         = misc.getCommitHash()

        super().accept()

    @gui.showExceptions
    def dataDirChanged(self, *args):
        self.loadSubjectIDs()
        self.loadParadigmFiles()
        self.updateDataFile()

    @gui.showExceptions
    def dataDirButtonClicked(self, *args):
        dataDir = self.dataDir.text()
        dataDir = gui.openDirectory(dataDir)

        if not dataDir:
            return

        dataDir = misc.absolutePath(dataDir)
        self.dataDir.setText(dataDir)
        self.dataDir.textEdited.emit(dataDir)

    @gui.showExceptions
    def autoDataFileChanged(self, state, *args):
        if state == QtCore.Qt.Checked:
            self.dataFile.setReadOnly(True)
            self.dataFileButton.setEnabled(False)
            self.updateDataFile()
        else:
            self.dataFile.setReadOnly(False)
            self.dataFileButton.setEnabled(True)

    @gui.showExceptions
    def dataFileButtonClicked(self, *args):
        file     = self.dataFile.text()
        dataDir  = self.dataDir.text()
        if file:
            file = misc.absolutePath(file, dataDir)
        else:
            file = dataDir
        file     = gui.saveFile(config.DATA_FILTER, file)

        if not file:
            return

        file = misc.relativePath(file, dataDir)
        self.dataFile.setText(file)

    @gui.showExceptions
    def calibrationFileButtonClicked(self, *args):
        file     = self.calibrationFile.text()
        dataDir  = self.dataDir.text()
        if file:
            file = misc.absolutePath(file, dataDir)
        else:
            file = dataDir
        file     = gui.openFile(config.SETTINGS_FILTER, file)

        if not file:
            return

        file = misc.relativePath(file, dataDir)
        self.calibrationFile.setText(file)

    @gui.showExceptions
    def paradigmFileChanged(self, text, *args):
        if text:
            self.rove.setEnabled(False)
        else:
            self.rove.setEnabled(True)

    @gui.showExceptions
    def paradigmFileButtonClicked(self, *args):
        file     = self.paradigmFile.currentText()
        dataDir  = self.dataDir.text()
        if file:
            file = misc.absolutePath(file, dataDir)
        else:
            file = dataDir
        file     = gui.openFile(config.SETTINGS_FILTER, file)

        if not file:
            return

        file = misc.relativePath(file, dataDir)
        self.paradigmFile.setCurrentText(file)

    @gui.showExceptions
    def okClicked(self, *args):
        gb.appMode = 'Experiment'
        self.accept()

    @gui.showExceptions
    def calibrationClicked(self, *args):
        gb.appMode = 'Calibration'
        self.accept()

    # def revertClicked(self, *args):
    #     self.dataDir.setText(gb.session.dataDir.value)
    #     self.loadSubjectIDs()
    #     self.subjectID.setCurrentText(gb.session.subjectID.value)
    #     self.experimentName.setCurrentText(gb.session.experimentName.value)
    #     self.experimentMode.setCurrentText(gb.session.experimentMode.value)
    #     self.recording.setCurrentText(gb.session.recording.value)
    #
    #     if gb.session.autoDataFile.value:
    #         self.autoDataFile.setCheckState(QtCore.Qt.Checked)
    #         self.updateDataFile()
    #         self.dataFile.setReadOnly(True)
    #         self.dataFileButton.setEnabled(False)
    #     else:
    #         self.autoDataFile.setCheckState(QtCore.Qt.Unchecked)
    #         self.dataFile.setText(gb.session.dataFile.value)
    #         self.dataFile.setCursorPosition(0)
    #         self.dataFile.setReadOnly(False)
    #         self.dataFileButton.setEnabled(True)
    #
    #     self.calibrationFile.setText(gb.session.calibrationFile.value)
    #     self.loadParadigmFiles()
    #     self.paradigmFile.setCurrentText(gb.session.paradigmFile.value)
    #
    #     if self.paradigmFile.currentText(): self.rove.setEnabled(False)
    #     else                              : self.rove.setEnabled(True )
    #
    #     for i in range(self.rove.count()):
    #         if self.roveParams[i] in gb.session.rove.value:
    #             checkState = QtCore.Qt.Checked
    #         else:
    #             checkState = QtCore.Qt.Unchecked
    #         self.rove.item(i).setCheckState(checkState)

    @gui.showExceptions
    def loadSubjectIDs(self, *args):
        dataDir = self.dataDir.text()
        subjectID = self.subjectID.currentText()
        self.subjectID.clear()
        if os.path.isdir(dataDir):
            try:
                subjectIDs = next(os.walk(dataDir))[1]
                self.subjectID.addItems(subjectIDs)
            except:
                pass
        self.subjectID.setCurrentText(subjectID)

    @gui.showExceptions
    def loadParadigmFiles(self, *args):
        dataDir      = self.dataDir.text()
        paradigmFile = self.paradigmFile.currentText()
        self.paradigmFile.clear()
        if os.path.isdir(dataDir):
            files = next(os.walk(dataDir))[2]
            paradigmFiles = ['']
            for file in files:
                if 'calibration' in file.lower():
                    continue
                fileName, fileExt = os.path.splitext(file)
                if fileExt == config.SETTINGS_EXT:
                    paradigmFiles += [file.replace('\\','/')]
            self.paradigmFile.addItems(paradigmFiles)
        self.paradigmFile.setCurrentText(paradigmFile)

    @gui.showExceptions
    def updateDataFile(self, *args):
        if self.autoDataFile.checkState() == QtCore.Qt.Checked:
            dataFile = os.path.join(
                # self.dataDir.text(),
                self.subjectID.currentText(),
                '%s-%s-%s-%s.h5' % (
                    self.subjectID.currentText(),
                    gb.sessionTimeCompact,
                    self.experimentName.currentText(),
                    self.recording.currentText() )
                ).replace('\\','/')
            self.dataFile.setText(dataFile)
            self.dataFile.setCursorPosition(0)
