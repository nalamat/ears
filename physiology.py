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

import gui
import daqs
import misc
import hdf5
import config
import plotting
import globals     as gb


log = logging.getLogger(__name__)


class PhysiologyWindow(QtWidgets.QMainWindow):

    ########################################
    # initialize the GUI

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
        self.addToolBar(QtCore.Qt.LeftToolBarArea , self.initControlBar()   )
        self.addToolBar(QtCore.Qt.LeftToolBarArea , self.initSettingsBar())
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
        # daqs.digitalInput.edgeDetected.connect(self.digitalInputEdgeDetected)
        daqs.physiologyInput.dataAcquired.connect(
            self.physiologyInputDataAcquired)

    def initPlot(self):
        physiologyFS          = daqs.physiologyInput.fs
        lineCount             = daqs.physiologyInput.lineCount
        lineLabels            = list(map(str, np.arange(lineCount)+1))
        behavior              = gb.behaviorWindow

        self.timePlot         = plotting.ChannelPlotWidget(xRange=10,
                                yLimits=(-4,15), yLabel='Electrodes',
                                yGrid=[-3.5,-1.5,-1] + list(range(lineCount)))

        self.physiologyTrace  = plotting.AnalogChannel(physiologyFS,
                                self.timePlot, label=lineLabels,
                                hdf5Node='/trace/physiology',
                                lineCount=lineCount, filter=(300,6e3),
                                yScale=.1, yOffset=14, yGap=-1, grandMean=True)

        # rectangular plots
        self.trialEpoch       = plotting.RectEpochChannel(self.timePlot,
                                label='Trial', source=behavior.trialEpoch,
                                yOffset=-2, yRange=.5, color=config.COLOR_TRIAL)
        self.targetEpoch      = plotting.RectEpochChannel(self.timePlot,
                                label='Target', source=behavior.targetEpoch,
                                yOffset=-2.5, yRange=.5,
                                color=config.COLOR_TARGET)
        self.pumpEpoch        = plotting.RectEpochChannel(self.timePlot,
                                label='Pump', source=behavior.pumpEpoch,
                                yOffset=-3, yRange=.5, color=config.COLOR_PUMP)
        self.timeoutEpoch     = plotting.RectEpochChannel(self.timePlot,
                                label='Timeout', source=behavior.timeoutEpoch,
                                yOffset=-3.5, yRange=.5,
                                color=config.COLOR_TIMEOUT)

        self.timePlot.timeBase = self.physiologyTrace
        self.timePlot.start()

        self.fftPlot          = plotting.ChannelPlotWidget(
                                xLimits=(0, 5e3), xGrid=1e3,
                                xLabel='Frequency (kHz)',
                                xTicksFormat=lambda x: '%g' % (x/1e3),
                                yLimits=(-4,15),
                                yGrid=[-3.5,-1.5,-1] + list(range(lineCount)),
                                timePlot=False)

        self.spikePlot        = plotting.ChannelPlotWidget(
                                xLimits=(-5e-3, 5e-3), xGrid=1e-3,
                                xLabel='Time (ms)',
                                xTicksFormat=lambda x: '%g' % (x*1e3),
                                yLimits=(-4,15),
                                yGrid=[-3.5,-1.5,-1] + list(range(lineCount)),
                                timePlot=False)

        layout = QtWidgets.QGridLayout()
        layout.setColumnStretch(0,2)
        # layout.setColumnStretch(1,1)
        # layout.setColumnStretch(2,1)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addWidget(self.timePlot  , 0, 0)
        # layout.addWidget(self.fftPlot   , 0, 1)
        # layout.addWidget(self.spikePlot , 0, 2)

        frame = QtWidgets.QFrame()
        frame.setLayout(layout)

        return frame

    def initControlBar(self):
        title = QtWidgets.QLabel('Control')
        title.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(1)

        # list of buttons to be generated for the control bar
        # '|' will define a vertical separator line
        buttonList = [
            ('apply'  , 'Apply'  ),
            ('revert' , 'Revert' ),
            ('load'   , 'Load'   ),
            ('save'   , 'Save'   ),
            ]

        self.buttons = misc.Dict()

        for (name, label) in buttonList:
            if name == '|':
                # add vertical separator line
                layout.addWidget(gui.QVSeparator())
            else:
                # generate and add button
                btn = QtWidgets.QToolButton()
                btn.setText(label)
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

        # layout.addStretch()

        frame = QtWidgets.QFrame()
        frame.setLayout(layout)

        bar = QtWidgets.QToolBar('Control')
        bar.setMovable(False)
        bar.setStyleSheet('.QToolBar{border-bottom:%s; border-right:%s}' %
            (config.BORDER_STYLE, config.BORDER_STYLE) )
        bar.addWidget(frame)

        return bar

    def initSettingsBar(self):
        title = QtWidgets.QLabel('Plot settings')
        title.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtWidgets.QVBoxLayout()

        yScale = int(20*np.log10(self.physiologyTrace.yScale))
        self.lblYScale = QtWidgets.QLabel('Scale: %d dB' % yScale)
        layout.addWidget(self.lblYScale)
        self.sldYScale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldYScale.setMinimum(-50)
        self.sldYScale.setMaximum(30)
        self.sldYScale.setSingleStep(5)
        self.sldYScale.setPageStep(5)
        self.sldYScale.setValue(yScale)
        self.sldYScale.valueChanged.connect(self.sldYScaleValueChanged)
        layout.addWidget(self.sldYScale)

        filterFL = int(self.physiologyTrace.filter[0])
        self.lblFlFilter = QtWidgets.QLabel('Lower cutoff: %d Hz' % filterFL)
        layout.addWidget(self.lblFlFilter)
        self.sldFilterFL = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldFilterFL.setMinimum(0)
        self.sldFilterFL.setMaximum(4e3)
        self.sldFilterFL.setSingleStep(100)
        self.sldFilterFL.setPageStep(100)
        self.sldFilterFL.setValue(filterFL)
        self.sldFilterFL.valueChanged.connect(self.sldFilterFLValueChanged)
        layout.addWidget(self.sldFilterFL)

        filterFH = int(self.physiologyTrace.filter[1])
        self.lblFhFilter = QtWidgets.QLabel('Higher cutoff: %d Hz' % filterFH)
        layout.addWidget(self.lblFhFilter)
        self.sldFilterFH = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldFilterFH.setMinimum(5e3)
        self.sldFilterFH.setMaximum(20e3)
        self.sldFilterFH.setSingleStep(200)
        self.sldFilterFH.setPageStep(200)
        self.sldFilterFH.setValue(filterFH)
        self.sldFilterFH.valueChanged.connect(self.sldFilterFHValueChanged)
        layout.addWidget(self.sldFilterFH)

        layout2 = QtWidgets.QVBoxLayout()
        layout2.addWidget(QtWidgets.QLabel('Left checkbox: Visible'))
        layout2.addWidget(QtWidgets.QLabel('Right checkbox: Grand mean'))
        # layout2.addWidget(QtWidgets.QLabel('Top row: Shanks'))
        # layout2.addWidget(QtWidgets.QLabel('Left column: Depths'))

        def makeCheckBoxes(label, callback, r, c):
            lbl = QtWidgets.QLabel(label)
            chks = (QtWidgets.QCheckBox(), QtWidgets.QCheckBox())
            layout4 = QtWidgets.QVBoxLayout()
            layout4.setAlignment(QtCore.Qt.AlignCenter)
            layout4.setSpacing(2)
            layout4.addWidget(lbl , 0, QtCore.Qt.AlignCenter)
            layout5 = QtWidgets.QHBoxLayout()
            for i in range(len(chks)):
                chks[i].setCheckState(QtCore.Qt.Checked)
                chks[i].stateChanged.connect(functools.partial(callback, i))
                layout5.addWidget(chks[i], 0, QtCore.Qt.AlignCenter)
            layout4.addLayout(layout5)
            layout3.addLayout(layout4, r, c)
            return chks

        layout3 = QtWidgets.QGridLayout()
        self.chkAll    = makeCheckBoxes('All', self.chkAllChanged, 0, 0)
        self.chkShanks = [None]*len(config.ELECTRODE_SHANKS)
        self.chkDepths = [None]*len(config.ELECTRODE_DEPTHS)
        self.chkElecs  = [None]*config.ELECTRODE_COUNT
        for c in range(len(config.ELECTRODE_SHANKS)):
            self.chkShanks[c] = makeCheckBoxes(config.ELECTRODE_SHANKS[c],
                functools.partial(self.chkShanksChanged, c), 0, c+1)
        for r in range(len(config.ELECTRODE_DEPTHS)):
            self.chkDepths[r] = makeCheckBoxes(config.ELECTRODE_DEPTHS[r],
                functools.partial(self.chkDepthsChanged, r), r+1, 0)
        for r in range(config.ELECTRODE_MAP.shape[0]):
            for c in range(config.ELECTRODE_MAP.shape[1]):
                elec = config.ELECTRODE_MAP.iloc[r, c]
                if not np.isnan(elec):
                    elec = int(elec)
                    self.chkElecs[elec-1] = makeCheckBoxes(str(elec),
                        functools.partial(self.chkElecsChanged, elec-1),
                        r+1, c+1,)
        layout2.addLayout(layout3)

        grp = QtWidgets.QGroupBox()
        grp.setTitle('Electrodes')
        grp.setLayout(layout2)

        layout.addWidget(grp)
        layout.addStretch()

        frame = QtWidgets.QFrame()
        frame.setLayout(layout)

        scroll = QtWidgets.QScrollArea()
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
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

    ########################################
    # GUI callbacks

    def closeEvent(self, event):
        if event.spontaneous():
            log.debug('Ignoring spontaneous close event on physiology window')
            event.ignore()
            return
        # log.debug('Stopping physiology plot')
        # self.plot.stop()
        # event.accept()
        # log.debug('Stopped physiology plot')

    @gui.showExceptions
    def applyClicked(self, *args):
        # get values from widgets
        self.physiologyTrace.refreshBlock   = True
        self.physiologyTrace.yScale         = 10**(self.sldYScale.value()/20)
        self.physiologyTrace.filter         = (self.sldFilterFL.value(),
           self.sldFilterFH.value())
        self.physiologyTrace.linesVisible   = tuple(
            chk[0].checkState()==QtCore.Qt.Checked for chk in self.chkElecs)
        self.physiologyTrace.linesGrandMean = tuple(
            chk[1].checkState()==QtCore.Qt.Checked for chk in self.chkElecs)
        self.physiologyTrace.refreshBlock = False
        self.physiologyTrace.refresh()

    @gui.showExceptions
    def sldYScaleValueChanged(self, value):
        self.lblYScale  .setText('Scale: %d dB'         % value)

    @gui.showExceptions
    def sldFilterFLValueChanged(self, value):
        self.lblFlFilter.setText('Lower cutoff: %d Hz'  % value)

    @gui.showExceptions
    def sldFilterFHValueChanged(self, value):
        self.lblFhFilter.setText('Higher cutoff: %d Hz' % value)

    @gui.showExceptions
    def chkAllChanged(self, index, *args):
        state = self.chkAll[index].checkState()
        for chk in self.chkShanks:
            gui.setCheckState(chk[index], state)
        for chk in self.chkDepths:
            gui.setCheckState(chk[index], state)
        for chk in self.chkElecs:
            gui.setCheckState(chk[index], state)

    @gui.showExceptions
    def chkShanksChanged(self, shank, index, *args):
        state = self.chkShanks[shank][index].checkState()
        for elec in config.ELECTRODE_MAP.iloc[:,shank]:
            if np.isnan(elec): continue
            gui.setCheckState(self.chkElecs[elec-1][index], state)
        self.refreshCheckBoxes()

    @gui.showExceptions
    def chkDepthsChanged(self, depth, index, *args):
        state = self.chkDepths[depth][index].checkState()
        for elec in config.ELECTRODE_MAP.iloc[depth,:]:
            if np.isnan(elec): continue
            gui.setCheckState(self.chkElecs[elec-1][index], state)
        self.refreshCheckBoxes()

    @gui.showExceptions
    def chkElecsChanged(self, elec, index, *args):
        self.refreshCheckBoxes()

    @gui.showExceptions
    def updateGUI(self):
        '''Called by `updateTimer` every 50ms'''
        # settings.status.ts .value = '%.2f' % self.getTS()
        # settings.status.fps.value = '%.1f' % self.plot.calculatedFPS
        pass

    ########################################
    # DAQ callbacks

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


    ########################################
    # house keeping

    def refreshCheckBoxes(self):
        def getState(elecs):
            for elec in elecs:
                if np.isnan(elec): continue
                chk = self.chkElecs[elec-1][index]
                if chk.checkState() == QtCore.Qt.Unchecked:
                    return QtCore.Qt.Unchecked
            return QtCore.Qt.Checked

        for index in range(2):
            gui.setCheckState(self.chkAll[index],
                getState(config.ELECTRODE_MAP.values.flatten()))
            for shank in range(len(self.chkShanks)):
                gui.setCheckState(self.chkShanks[shank][index],
                    getState(config.ELECTRODE_MAP.iloc[:,shank]))
            for depth in range(len(self.chkDepths)):
                gui.setCheckState(self.chkDepths[depth][index],
                    getState(config.ELECTRODE_MAP.iloc[depth,:]))


# playground
if __name__ == '__main__':
    @gui.showExceptions
    def do():
        raise ValueError('something')

    do()



# end
