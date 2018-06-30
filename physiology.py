'''Physiology window for viewing neural activity in real-time.


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
import copy
import json
import time
import logging
import platform
import functools
import scipy.stats
import numpy       as     np
from   PyQt5       import QtCore, QtWidgets, QtGui

import daqs
import misc
import hdf5
import config
import plotting
import guiHelper
import globals     as gb


log = logging.getLogger(__name__)


class PhysiologyWindow(QtWidgets.QMainWindow):
    # initializing the interface
    def __init__(self, parent=None):
        super().__init__(parent)

        self.initDAQ()

        # init central section of GUI
        central = QtWidgets.QMainWindow()
        # central.addToolBar(QtCore.Qt.TopToolBarArea, self.initControlBar())
        # central.addToolBar(QtCore.Qt.BottomToolBarArea,
        #     self.initPerformanceBar())
        central.setCentralWidget(self.initPlot())

        # init side bars
        # self.addToolBar(QtCore.Qt.LeftToolBarArea , self.initSessionBar() )
        # self.addToolBar(QtCore.Qt.LeftToolBarArea , self.initRoveBar()    )
        # self.addToolBar(QtCore.Qt.LeftToolBarArea , self.initParadigmBar())
        # self.addToolBar(QtCore.Qt.RightToolBarArea, self.initStatusBar()  )
        # self.addToolBar(QtCore.Qt.RightToolBarArea, self.initTrialLogBar())

        # init the window
        self.setCentralWidget(central)
        self.setWindowTitle(config.APP_NAME + ' Physiology')
        self.setWindowIcon(QtGui.QIcon(config.APP_LOGO))
        self.setStyleSheet('.QToolBar,.QScrollArea{border:0px}')
        self.setAttribute(QtCore.Qt.WA_QuitOnClose, False)

        # init update timer (for time varying GUI elements)
        # self.updateTimer       = QtCore.QTimer()
        # self.updateTimer.timeout.connect(self.updateGUI)
        # self.updateTimer.setInterval(10)    # every 50 ms

        # self.forceTrialType = 'Go remind'
        # self.evaluateParadigm()

    def initDAQ(self):
        daqs.digitalInput.edgeDetected.connect(self.digitalInputEdgeDetected)
        daqs.physiologyInput.dataAcquired.connect(
            self.physiologyInputDataAcquired)

    def initPlot(self):
        lineCount            = daqs.physiologyInput.lineCount
        self.plot            = plotting.ChannelPlotWidget(yLimits=(-.5, 15.5),
                               yGrid=list(range(lineCount)))
        self.physiologyTrace = plotting.AnalogChannel(daqs.physiologyInput.fs,
                               '/trace/physiology', self.plot, lineCount,
                               yScale=.05, timeBase=True)
        self.plot.start()

        # self.generator    = plotting.AnalogGenerator(self.pokeTrace,
        #                     self.spoutTrace, self.speakerTrace, self.micTrace)

        return self.plot

    def sldYScaleValueChanged(self, value):
        self.lblYScale.setText('%d dB' % value)
        self.physiologyTrace.yScale = 10**(value/20)

    def sldFlFilterValueChanged(self, value):
        self.lblFlFilter.setText('%d Hz' % value)
        self.physiologyTrace.flFilter = value

    def sldFhFilterValueChanged(self, value):
        self.lblFhFilter.setText('%d Hz' % value)
        self.physiologyTrace.fhFilter = value

    def initParadigmBar(self):
        title = QtWidgets.QLabel('Paradigm settings')
        title.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtWidgets.QVBoxLayout()

        yScale = int(20*np.log10(self.physiologyTrace.yScale))
        self.lblYScale = QtWidgets.QLabel('%d dB' % yScale)
        layout.addWidget(self.lblYScale)
        self.sldYScale = QtWidgets.QSlider()
        self.sldYScale.setMinimum(-40)
        self.sldYScale.setMaximum(40)
        self.sldYScale.setValue(yScale)
        self.sldYScale.valueChanged.connect(self.sldYScaleValueChanged)
        layout.addWidget(self.sldYScale)

        flFilter = int(self.physiologyTrace.flFilter)
        self.lblFlFilter = QtWidgets.QLabel('%d Hz' % flFilter)
        layout.addWidget(self.lblFlFilter)
        self.sldFlFilter = QtWidgets.QSlider()
        self.sldFlFilter.setMinimum(0)
        self.sldFlFilter.setMaximum(4999)
        self.sldFlFilter.setValue(yScale)
        self.sldFlFilter.valueChanged.connect(self.sldFlFilterValueChanged)
        layout.addWidget(self.sldFlFilter)

        fhFilter = int(self.physiologyTrace.fhFilter)
        self.lblFhFilter = QtWidgets.QLabel('%d Hz' % fhFilter)
        layout.addWidget(self.lblFhFilter)
        self.sldFhFilter = QtWidgets.QSlider()
        self.sldFhFilter.setMinimum(0)
        self.sldFhFilter.setMaximum(4999)
        self.sldFhFilter.setValue(yScale)
        self.sldFhFilter.valueChanged.connect(self.sldFhFilterValueChanged)
        layout.addWidget(self.sldFhFilter)


        # for item in settings.paradigm:
        #     if item.name == 'group':
        #         layout2 = QtWidgets.QGridLayout()
        #         layout2.setColumnStretch(0,1)
        #         layout2.setColumnStretch(1,1)
        #         grp = QtWidgets.QGroupBox()
        #         grp.setTitle(item.label)
        #         grp.setLayout(layout2)
        #         layout.addWidget(grp)
        #
        #     elif item.name == 'rove':
        #         # rove table is generated in `self.initRoveBar`
        #         pass
        #
        #     elif item.name not in settings.paradigm.rove.value.keys():
        #         wig2 = None
        #
        #         if item.type in (int, float, str) and item.values:
        #             wig = QtWidgets.QComboBox()
        #             wig.addItems(item.values)
        #             wig.setCurrentText(item.value)
        #             signal = wig.currentIndexChanged
        #         elif item.type in (int, float, str) and not item.values:
        #             wig = QtWidgets.QLineEdit(str(item.value))
        #             signal = wig.textEdited
        #             if item.name in ['targetFile', 'maskerFile']:
        #                 # make a small button for open file dialog
        #                 h = wig.sizeHint().height()
        #                 wig2 = guiHelper.makeFileButton('Select file')
        #                 wig2.clicked.connect(
        #                     functools.partial(self.paradigmButtonClicked, item))
        #         elif item.type is bool:
        #             wig = QtWidgets.QCheckBox()
        #             if item.value:
        #                 wig.setCheckState(QtCore.Qt.Checked)
        #             else:
        #                 wig.setCheckState(QtCore.Qt.Unchecked)
        #             signal = wig.stateChanged
        #         else:
        #             raise ValueError('Unexpected parameter type')
        #
        #         signal.connect(functools.partial(self.paradigmChanged, item))
        #         item.widget  = wig
        #
        #         lbl = QtWidgets.QLabel(item.label + ':')
        #         layout2.addWidget(lbl, layout2.rowCount(), 0)
        #
        #         if wig2 is None:
        #             layout2.addWidget(wig, layout2.rowCount()-1, 1)
        #         else:
        #             item.widget2 = wig2
        #             itemLayout = QtWidgets.QHBoxLayout()
        #             itemLayout.addWidget(wig)
        #             itemLayout.addWidget(wig2)
        #             layout2.addLayout(itemLayout, layout2.rowCount()-1, 1)

        frame = QtWidgets.QFrame()
        frame.setLayout(layout)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True);
        scroll.setWidget(frame)
        scroll.setStyleSheet('.QScrollArea{border:%s}' % config.BORDER_STYLE)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(scroll)

        frame = QtWidgets.QFrame()
        frame.setLayout(layout)

        bar = QtWidgets.QToolBar(title.text())
        bar.setMovable(False)
        bar.setStyleSheet('.QToolBar{border-right:%s}' % config.BORDER_STYLE)
        bar.addWidget(frame)

        return bar

    # GUI callbacks
    def closeEvent(self, event):
        if event.spontaneous:
            event.ignore()
            return

    @guiHelper.showExceptions
    def paradigmChanged(self, item):
        # if item.name == 'targetType':
        #     tone = item.getWidgetValue() == 'Tone'
        #     if settings.paradigm.targetFrequency.widget:
        #         settings.paradigm.targetFrequency.widget.setEnabled(tone)
        #     if settings.paradigm.targetFile.widget:
        #         settings.paradigm.targetFile.widget .setEnabled(not tone)
        #         settings.paradigm.targetFile.widget2.setEnabled(not tone)

        if settings.status.experimentState.value == 'Not started':
            item.value = item.getWidgetValue()

        else:
            hasChanged = False
            for item in settings.paradigm:
                if item.widget and item.value != item.getWidgetValue():
                    hasChanged = True
                    break

            self.buttons.apply .setEnabled(hasChanged)
            self.buttons.revert.setEnabled(hasChanged)

    @guiHelper.showExceptions
    def paradigmButtonClicked(self, item):
        if item.name in ['targetFile', 'maskerFile']:
            file = item.getWidgetValue()
            if not file: file = config.STIM_DIR
            file = guiHelper.openFile('Sound (*.wav)', file)
            if file:
                item.widget.setText(file.replace('\\','/'))
                self.paradigmChanged(item)

    @guiHelper.showExceptions
    def applyClicked(self, *args):
        # trasnfer widget values to paradigm
        for item in settings.paradigm:
            if item.widget:
                item.value = item.getWidgetValue()

        # evaluated right away if no trials ongoing
        if settings.status.trialState.value == 'Poke start':
            self.evaluateParadigm()

        self.buttons.apply .setEnabled(False)
        self.buttons.revert.setEnabled(False)

    @guiHelper.showExceptions
    def revertClicked(self, *args):
        # transfer paradigm values to widgets
        for item in settings.paradigm:
            if item.widget:
                item.setWidgetValue(item.value)

        self.buttons.apply .setEnabled(False)
        self.buttons.revert.setEnabled(False)

    @guiHelper.showExceptions
    def loadClicked(self, *args):
        file = guiHelper.openFile(config.SETTINGS_FILTER,
            settings.session.paradigmFile.value)

        if not file: return

        settings.paradigm.loadFile(file)
        settings.session.paradigmFile.value = file

        for item in settings.paradigm:
            if item.widget:
                item.setWidgetValue(item.value)

        # evaluated right away if no trials ongoing
        if settings.status.trialState.value == 'Poke start':
            self.evaluateParadigm()

        self.buttons.apply .setEnabled(False)
        self.buttons.revert.setEnabled(False)

    @guiHelper.showExceptions
    def saveClicked(self, *args):
        file = guiHelper.saveFile(config.SETTINGS_FILTER,
            settings.session.paradigmFile.value)

        if file:
            settings.paradigm.saveFile(file)
            settings.session.paradigmFile.value = file

    @guiHelper.showExceptions
    def updateGUI(self):
        '''Called by `updateTimer` every 50ms'''
        # settings.status.ts .value = '%.2f' % self.getTS()
        # settings.status.fps.value = '%.1f' % self.plot.calculatedFPS
        pass

    # IO callbacks
    def digitalInputEdgeDetected(self, task, name, edge, time):
        # log.info('Detected %s edge on %s at %.3f', edge, name, time)
        # channels = {'poke':self.pokeEpoch, 'spout':self.spoutEpoch,
        #     'button':self.buttonEpoch}
        # if name in channels:
        #     if edge == 'rising':
        #         channels[name].append(start=time)
        #     elif edge == 'falling':
        #         channels[name].append(stop=time)
        pass

    def physiologyInputDataAcquired(self, task, data):
        self.physiologyTrace.append(data)


# playground
if __name__ == '__main__':
    @guiHelper.showExceptions
    def do():
        raise ValueError('something')

    do()



# end
