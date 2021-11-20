'''Physiology window for viewing neural activity in real-time.


This file is part of the EARS project <https://github.com/nalamat/ears>
Copyright (C) 2017-2021 Nima Alamatsaz <nima.alamatsaz@gmail.com>
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
import pypeline
import globals     as gb
import glPlotLib   as glp


log = logging.getLogger(__name__)


class PhysiologyWindow(QtWidgets.QMainWindow):

    ########################################
    # initialize the GUI

    def __init__(self, parent=None):
        super().__init__(parent)

        self.initDAQ()
        self.initStorage()

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
        # self.updateTimer.setInterval(1000)    # every 50 ms
        # self.updateTimer.start()

        # self.forceTrialType = 'Go remind'
        # self.evaluateParadigm()

    def initDAQ(self):
        # daqs.digitalInput.edgeDetected.connect(self.digitalInputEdgeDetected)
        daqs.physiologyInput.dataAcquired.connect(
            self.physiologyInputDataAcquired)

    def initStorage(self):
        self.physiologyStorage = hdf5.AnalogStorage('/trace/physiology')

        # storage pipeline
        daqs.physiologyInput >> self.physiologyStorage

    def initPlot(self):
        physiologyFS           = daqs.physiologyInput.fs
        channelCount           = daqs.physiologyInput.channelCount
        lineLabels             = list(map(str, np.arange(channelCount)+1))
        behavior               = gb.behaviorWindow
        tsRange                = 10

        div = 1
        while (channelCount / div) % 1 != 0 or channelCount / div > 16:
            div += 1
        yTickCount = int(channelCount / div + 1)

        #
        self.canvas = glp.Canvas(self)
        self.scope = glp.Scope(self.canvas, pos=(0,0), size=(1,1),
            margin=(60,20,20,10), tsRange=tsRange,
            xTicks=np.linspace(0, 1, tsRange+1),
            yTicks=np.linspace(.1, 1, yTickCount))

        # analog physiology plot
        self.physiologyPlot = glp.AnalogPlot(self.scope, pos=(0,.1),
            size=(1,.9), label=np.arange(channelCount)+1,
            fgColor=glp.defaultColors, fontSize=18)
        # self.spikeOverlay = SpikeOverlay(self.scope, pos=(0,.1), size=(1,.9))

        # physiology processing nodes
        self.grandAverage      = pypeline.GrandAverage()
        self.filter            = pypeline.LFilter(fl=300, fh=6e3, n=6)
        self.scaler            = pypeline.Scaler(scale=0, dB=True)

        # processing and plotting pypeline
        (daqs.physiologyInput >> pypeline.Thread() >>
            self.grandAverage >> self.filter >> self.scaler >>
            (self.scope,
            self.physiologyPlot))
            # >> pypeline.Split() >> (pypeline.Node(), pypeline.DummySink(14))
            # >> (self.physiologyTrace0,
            #    pypeline.DownsampleMinMax(ds=50) >> self.physiologyTrace1,
            #    pypeline.DownsampleLTTB(fsOut=31250/50) >> self.physiologyTrace2,
            #    pypeline.DownsampleAverage(ds=50) >> self.physiologyTrace3,
            #    ))

        # rectangular epoch plots
        self.trialEpochPlot = glp.EpochPlot(self.scope, pos=(0,.075),
            size=(1,.025), label='Trial', fgColor=config.COLOR_TRIAL)
        self.targetEpochPlot = glp.EpochPlot(self.scope, pos=(0,.05),
            size=(1,.025), label='Target', fgColor=config.COLOR_TARGET)
        self.pumpEpochPlot = glp.EpochPlot(self.scope, pos=(0,.025),
            size=(1,.025), label='Pump', fgColor=config.COLOR_PUMP)
        self.timeoutEpochPlot = glp.EpochPlot(self.scope, pos=(0,0),
            size=(1,.025), label='Timeout', fgColor=config.COLOR_TIMEOUT)

        behavior.trialEpochPlot   >> self.trialEpochPlot
        behavior.targetEpochPlot  >> self.targetEpochPlot
        behavior.pumpEpochPlot    >> self.pumpEpochPlot
        behavior.timeoutEpochPlot >> self.timeoutEpochPlot

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        # layout.addWidget(self.timePlot  , 0, 0)
        # layout.addWidget(self.fftPlot   , 0, 1)
        # layout.addWidget(self.spikePlot , 0, 2)
        layout.addWidget(self.canvas)

        # for i in range(4):
        #     for j in range(4):
        #         for k in range(3):
        #             plot = plotting.ScrollingPlotWidget(timePlot=False,
        #                 xLimits=(-5e-3, 5e-3), xGrid=1e-3, xLabel='Time (ms)',
        #                 xTicksFormat=lambda x: '%g' % (x*1e3),
        #                 yLimits=(-2e-3,2e-3), yGrid=.5e-3)
        #             # plot.layout().setContentsMargins(0,0,0,0)
        #             plot.setLabel('left','')
        #             plot.setLabel('top','<br/><br/><br/><br/><br/>')
        #             plot.setLabel('top','')
        #             plot.getAxis('left').setWidth(0)
        #             plot.getAxis('top').setHeight(0)
        #             plot.getAxis('right').setWidth(0)
        #             plot.getAxis('bottom').setHeight(0)
        #             if k == 0:
        #                 layout.addWidget(plot, i*2, j*2, 1, 2)
        #             elif k == 1:
        #                 layout.addWidget(plot, i*2+1, j*2)
        #             elif k == 2:
        #                 layout.addWidget(plot, i*2+1, j*2+1)

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

        self.lblYScale = QtWidgets.QLabel('Scale: %d dB' % self.scaler.scale)
        layout.addWidget(self.lblYScale)
        self.sldYScale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldYScale.setMinimum(-60)
        self.sldYScale.setMaximum(60)
        self.sldYScale.setSingleStep(5)
        self.sldYScale.setPageStep(5)
        self.sldYScale.setValue(self.scaler.scale)
        self.sldYScale.valueChanged.connect(self.sldYScaleValueChanged)
        layout.addWidget(self.sldYScale)

        # filterFL = int(self.physiologyTrace.filter[0])
        filterFL = int(self.filter.fl)
        self.lblFlFilter = QtWidgets.QLabel('Lower cutoff: %d Hz' % filterFL)
        layout.addWidget(self.lblFlFilter)
        self.sldFilterFL = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldFilterFL.setMinimum(0)
        self.sldFilterFL.setMaximum(2e3)
        self.sldFilterFL.setSingleStep(100)
        self.sldFilterFL.setPageStep(100)
        self.sldFilterFL.setValue(filterFL)
        self.sldFilterFL.valueChanged.connect(self.sldFilterFLValueChanged)
        layout.addWidget(self.sldFilterFL)

        # filterFH = int(self.physiologyTrace.filter[1])
        filterFH = int(self.filter.fh)
        self.lblFhFilter = QtWidgets.QLabel('Higher cutoff: %d Hz' % filterFH)
        layout.addWidget(self.lblFhFilter)
        self.sldFilterFH = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldFilterFH.setMinimum(2e3)
        self.sldFilterFH.setMaximum(15e3)
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
        # self.physiologyTrace.refreshBlock   = True
        # yScale = 10**(self.sldYScale.value()/20)
        # self.physiologyTrace.yScale = yScale
        # self.physiologyTrace0.yScale = yScale
        # self.physiologyTrace1.yScale = yScale
        # self.physiologyTrace2.yScale = yScale
        # self.physiologyTrace3.yScale = yScale
        # self.physiologyTrace.filter         = (self.sldFilterFL.value(),
        #    self.sldFilterFH.value())
        # self.physiologyTrace.linesVisible   = tuple(
        #     chk[0].checkState()==QtCore.Qt.Checked for chk in self.chkElecs)
        # self.physiologyTrace.linesGrandMean = tuple(
        #     chk[1].checkState()==QtCore.Qt.Checked for chk in self.chkElecs)

        # self.physiologyTrace.refreshBlock = False
        # self.physiologyTrace.refresh()
        pass

    @gui.showExceptions
    def sldYScaleValueChanged(self, value):
        self.lblYScale.setText('Scale: %d dB'         % value)
        self.scaler.scale = value

    @gui.showExceptions
    def sldFilterFLValueChanged(self, value):
        self.lblFlFilter.setText('Lower cutoff: %d Hz'  % value)
        self.filter.fl = self.sldFilterFL.value()

    @gui.showExceptions
    def sldFilterFHValueChanged(self, value):
        self.lblFhFilter.setText('Higher cutoff: %d Hz' % value)
        self.filter.fh = self.sldFilterFH.value()

    @gui.showExceptions
    def chkAllChanged(self, index, *args):
        state = self.chkAll[index].checkState()
        for chk in self.chkShanks:
            gui.setCheckState(chk[index], state)
        for chk in self.chkDepths:
            gui.setCheckState(chk[index], state)
        for chk in self.chkElecs:
            gui.setCheckState(chk[index], state)
        self.refreshCheckBoxes()

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
        # settings.status.fps.value = '%.1f' % self.plot.measuredFPS
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
        # self.physiologyTrace.append(data)
        # self.fftPlot.append(data)
        # for i in range(len(self.traces)):
        #     self.traces[i].append(data[i,:])
        pass

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

        self.grandAverage.mask = [chk[1].checkState()==QtCore.Qt.Checked
            for chk in self.chkElecs]


# playground
if __name__ == '__main__':
    @gui.showExceptions
    def do():
        raise ValueError('something')

    do()



# end
