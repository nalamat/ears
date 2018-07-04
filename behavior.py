'''Behavior window for closed-loop control of experiment paradigms.


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
import time
import queue
import logging
import platform
import functools
import threading
import traceback
import scipy.stats
import scipy.io.wavfile
import numpy            as     np
import scipy            as     sp
import datetime         as     dt
from   PyQt5            import QtCore, QtWidgets, QtGui

import daq
import daqs
import hdf5
import misc
import pump
import config
import plotting
import guiHelper
import globals          as     gb


log = logging.getLogger(__name__)


class BehaviorWindow(QtWidgets.QMainWindow):

    runInGUIThreadSignal = QtCore.pyqtSignal(object, object, object, object)

    ########################################
    # initialize the GUI

    def __init__(self, parent=None):
        super().__init__(parent)

        self.initLocals()
        self.initGlobals()    # before init bars
        self.initDAQ()
        self.initPump()

        # init central section of GUI
        central = QtWidgets.QMainWindow()
        central.addToolBar(QtCore.Qt.TopToolBarArea, self.initControlBar())
        central.addToolBar(QtCore.Qt.BottomToolBarArea,
            self.initPerformanceBar())
        central.setCentralWidget(self.initPlot())

        # init side bars
        self.addToolBar(QtCore.Qt.LeftToolBarArea , self.initSessionBar    ())
        self.addToolBar(QtCore.Qt.LeftToolBarArea , self.initRoveBar       ())
        self.addToolBar(QtCore.Qt.LeftToolBarArea , self.initParadigmBar   ())
        self.addToolBar(QtCore.Qt.RightToolBarArea, self.initStatusBar     ())
        if config.SIM:
            self.addToolBar(QtCore.Qt.RightToolBarArea,
                self.initSoftControlBar())
        self.addToolBar(QtCore.Qt.RightToolBarArea, self.initTrialLogBar   ())

        # init the window
        self.setCentralWidget(central)
        self.setWindowTitle(config.APP_NAME + ' Behavior')
        self.setWindowIcon(QtGui.QIcon(config.APP_LOGO))
        self.setStyleSheet('.QToolBar,.QScrollArea{border:0px}')

    def initLocals(self):
        self.buttons = misc.Dict()

        self.soundData     = {'':None}
        self.targetFile    = ''       # local copy for detecting changes
        self.targetLevel   = 0        # local copy for detecting changes
        self.targetStartNS = 0
        self.maskerFile    = ''       # local copy for detecting changes
        self.maskerLevel   = 0        # local copy for detecting changes
        self.maskerStartNS = 0

        self.syringeDiameter = 0

        # self.trialActive   = False
        self.targetActive  = False
        self.pumpActive    = False
        self.pumpUpdateReq = False
        self.timeoutActive = False

        # update timer for time varying GUI elements
        self.updateTimer = QtCore.QTimer()
        self.updateTimer.timeout.connect(self.updateGUI)
        self.updateTimer.setInterval(20)    # every 20 ms

        # event handling
        self.eventQueue      = queue.Queue()
        self.eventLock       = threading.Lock()
        self.eventTimerStop  = None
        self.eventTimer      = None
        # self.handleEventTimer = QtCore.QTimer()
        # self.handleEventTimer.timeout.connect(self.handleEvent)
        # self.handleEventTimer.setInterval(5)    # every 5 ms
        self.eventThreadStop = threading.Event()
        self.eventThread     = None    # will init in `startEventThread`

        self.runInGUIThreadSignal.connect(self.runInGUIThreadCallback)
        self.guiThread = threading.current_thread()

        # always start the with a go remind trial
        self.forceTrialType = 'Go remind'
        gb.status.trialType.value = self.forceTrialType

    def initGlobals(self):
        # update trial and performance contexts with selected rove parameters
        for name in reversed(gb.paradigm.rove.keys):
            item = gb.trial[name]
            # remove units from label
            # note: gb.trial doesn't necessarily have labels
            label = gb.paradigm[name].shortLabel
            # replace white space with new line
            item.label = label.replace(' ', '\n')
            # remove and insert the item at the beginning of contexts
            gb.trial.remove(item)
            gb.trial.insert(0, item)
            gb.performance.insert(0, item.copy())

        # keey a local copy for widget interaction (apply/revert)
        self.paradigm = gb.paradigm.copy()
        # keey a local copy for verification and evaluation
        self.trial    = gb.trial   .copy()

        # init HDF5 and pandas.DataFrame logging
        gb.session    .initData  ('/log/session'    )
        gb.session    .appendData()
        gb.calibration.initData  ('/log/calibration', columnHeaders=False)
        gb.calibration.appendData()
        gb.paradigm   .initData  ('/log/paradigm'   , asString     =True )
        gb.trial      .initData  ('/log/trial'      )
        gb.performance.initData  ('/log/performance')

        log.info('Session context: %s'    , str(gb.session    ))
        log.info('Calibration context: %s', str(gb.calibration))

    def initDAQ(self):
        daqs.init()

        daqs.digitalInput.edgeDetected.connect(self.digitalInputEdgeDetected)
        daqs.analogInput .dataAcquired.connect(self.analogInputDataAcquired )
        daqs.analogOutput.dataNeeded  .connect(self.analogOutputDataNeeded  )

    def initPump(self):
        self.pump = pump.PumpInterface()
        self.pump.setDirection('infuse')

    def initPlot(self):

        inputFS           = daqs.analogInput.fs
        physiology        = gb.session.recording.value == 'Physiology'
        speakerNode       = '/trace/speaker' if physiology else None
        micNode           = '/trace/mic'     if physiology else None

        # plot widget
        self.plot         = plotting.ChannelPlotWidget(yLimits=(-1,7.5),
                            yGrid=[-.5,0,1,2,2.5,4.5,5,6,7], xRange=10)

        # analog traces
        self.speakerTrace = plotting.AnalogChannel(inputFS, self.plot,
                            label='Speaker', hdf5Node=speakerNode,
                            yScale=.1, yOffset=6.5, color=config.COLOR_SPEAKER)
        self.micTrace     = plotting.AnalogChannel(inputFS, self.plot,
                            label='Mic', hdf5Node=micNode,
                            yScale=.1, yOffset=5.5, color=config.COLOR_MIC)
        self.pokeTrace    = plotting.AnalogChannel(inputFS, self.plot,
                            label='Poke', labelOffset=.5,
                            yScale=.2, yOffset=1, color=config.COLOR_POKE)
        self.spoutTrace   = plotting.AnalogChannel(inputFS, self.plot,
                            label='Spout', labelOffset=.5,
                            yScale=.2, yOffset=0, color=config.COLOR_SPOUT)

        # rectangular epochs
        self.trialEpoch   = plotting.RectEpochChannel(self.plot,
                            label='Trial', hdf5Node='/epoch/trial',
                            yOffset=4, yRange=.5, color=config.COLOR_TRIAL)
        self.targetEpoch  = plotting.RectEpochChannel(self.plot,
                             label='Target', hdf5Node='/epoch/target',
                            yOffset=3.5, yRange=.5, color=config.COLOR_TARGET)
        self.pumpEpoch    = plotting.RectEpochChannel(self.plot,
                            label='Pump', hdf5Node='/epoch/pump',
                            yOffset=3, yRange=.5, color=config.COLOR_PUMP)
        self.timeoutEpoch = plotting.RectEpochChannel(self.plot,
                            label='Timeout', hdf5Node='/epoch/timeout',
                            yOffset=2.5, yRange=.5, color=config.COLOR_TIMEOUT)

        # symbol epochs
        self.pokeEpoch    = plotting.SymbEpochChannel(self.plot,
                            hdf5Node='/epoch/poke',
                            yOffset=1.5, color=config.COLOR_POKE)
        self.spoutEpoch   = plotting.SymbEpochChannel(self.plot,
                            hdf5Node='/epoch/spout',
                            yOffset=0.5, color=config.COLOR_SPOUT)
        self.buttonEpoch  = plotting.SymbEpochChannel(self.plot,
                            label='Button', hdf5Node='/epoch/button',
                            yOffset=-.25, color=config.COLOR_BUTTON)

        # self.generator    = plotting.AnalogGenerator(self.pokeTrace,
        #                     self.spoutTrace, self.speakerTrace, self.micTrace)

        self.plot.timeBase = self.speakerTrace

        return self.plot

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
            ('|'      , ''       ),
            ('start'  , 'Start'  ),
            ('pause'  , 'Pause'  ),
            ('go'     , 'Go'     ),
            ('nogo'   , 'Nogo'   ),
            ('|'      , ''       ),
            ('trial'  , 'Trial'  ),
            ('target' , 'Target' ),
            ('pump'   , 'Pump'   ),
            ('timeout', 'Timeout'),
            ('|'      , ''       ),
            ('export' , 'Export' ),
            ('close'  , 'Close'  ),
            ]

        for (name, label) in buttonList:
            if name == '|':
                # add vertical separator line
                layout.addWidget(guiHelper.QVSeparator())
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

        layout.addStretch()
        self.buttons.apply  .setEnabled(False)
        self.buttons.revert .setEnabled(False)
        self.buttons.pause  .setEnabled(False)
        self.buttons.trial  .setEnabled(False)
        self.buttons.target .setEnabled(False)
        self.buttons.pump   .setEnabled(False)
        self.buttons.timeout.setEnabled(False)
        self.buttons.export .setEnabled(False)

        frame = QtWidgets.QFrame()
        frame.setLayout(layout)

        bar = QtWidgets.QToolBar('Control')
        bar.setMovable(False)
        bar.setStyleSheet('.QToolBar{border-bottom:%s}' % config.BORDER_STYLE)
        bar.addWidget(frame)

        return bar

    def initSessionBar(self):
        title = QtWidgets.QLabel('Session settings')
        title.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtWidgets.QGridLayout()
        layout.setColumnStretch(0,1)
        layout.setColumnStretch(1,1)

        for item in gb.session:
            if item.name in ['subjectID', 'experimentName', 'experimentMode',
                    'recording']:
                lbl = QtWidgets.QLabel(item.label + ':')
                wig = QtWidgets.QLineEdit(item.value)
                wig.setReadOnly(True)
            elif item.name == 'dataViable':
                lbl = QtWidgets.QLabel(item.label + ':')
                wig = QtWidgets.QCheckBox()
                if item.value:
                    wig.setCheckState(QtCore.Qt.Checked)
                else:
                    wig.setCheckState(QtCore.Qt.Unchecked)
                callback = functools.partial(self.sessionChanged, item)
                wig.stateChanged.connect(callback)
            else:
                continue
            item.widget = wig
            layout.addWidget(lbl, layout.rowCount()  , 0)
            layout.addWidget(wig, layout.rowCount()-1, 1)

        frame = QtWidgets.QFrame()
        frame.setLayout(layout)
        frame.setStyleSheet('.QFrame{border:%s}' % config.BORDER_STYLE)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(frame)

        frame = QtWidgets.QFrame()
        frame.setLayout(layout)

        bar = QtWidgets.QToolBar(title.text())
        bar.setMovable(False)
        bar.setStyleSheet('.QToolBar{border-right:%s;border-bottom:%s}' %
            (config.BORDER_STYLE, config.BORDER_STYLE))
        bar.addWidget(frame)

        return bar

    def initRoveBar(self):
        title = QtWidgets.QLabel('Rove parameters')
        title.setAlignment(QtCore.Qt.AlignCenter)

        roveAdd = QtWidgets.QPushButton('Add')
        roveAdd.clicked.connect(self.roveAddClicked)
        roveRemove = QtWidgets.QPushButton('Remove')
        roveRemove.clicked.connect(self.roveRemoveClicked)

        labels = []
        for name in self.paradigm.rove.keys:
            labels += [self.paradigm[name].label]

        wig = QtWidgets.QTableWidget(1, len(labels))
        wig.setHorizontalHeaderLabels(labels)
        wig.horizontalHeader().setResizeMode(QtWidgets.QHeaderView.Stretch)
        wig.horizontalHeader().setHighlightSections(False)
        wig.verticalHeader().setVisible(False)
        wig.verticalHeader().setHighlightSections(False)
        wig.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        height = wig.verticalHeader().sizeHint().height()
        wig.setMaximumHeight(height*6)
        wig.cellChanged.connect(functools.partial(
            self.paradigmChanged, self.paradigm.rove))
        self.paradigm.rove.widget = wig
        self.paradigm.rove.getWidgetValue = functools.partial(
            self.roveGetWidgetValue, self.paradigm.rove)
        self.paradigm.rove.setWidgetValue = functools.partial(
            self.roveSetWidgetValue, self.paradigm.rove)
        self.paradigm.rove.setWidgetValue(self.paradigm.rove.value)

        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(roveAdd)
        layout2.addWidget(roveRemove)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(title)
        layout.addLayout(layout2)
        layout.addWidget(wig)

        frame = QtWidgets.QFrame()
        frame.setLayout(layout)

        bar = QtWidgets.QToolBar(title.text())
        bar.setMovable(False)
        bar.setStyleSheet('.QToolBar{border-right:%s;border-bottom:%s}' %
            (config.BORDER_STYLE,config.BORDER_STYLE))
        bar.addWidget(frame)

        return bar

    def initParadigmBar(self):
        title = QtWidgets.QLabel('Paradigm settings')
        title.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtWidgets.QVBoxLayout()

        for item in self.paradigm:
            if item.name == 'group':
                layout2 = QtWidgets.QGridLayout()
                layout2.setColumnStretch(0,1)
                layout2.setColumnStretch(1,1)
                grp = QtWidgets.QGroupBox()
                grp.setTitle(item.label)
                grp.setLayout(layout2)
                layout.addWidget(grp)

            elif item.name == 'rove':
                # rove table is generated in `self.initRoveBar`
                pass

            elif item.name not in self.paradigm.rove.keys:
                wig2 = None

                if item.type in (int, float, str) and item.values:
                    wig = QtWidgets.QComboBox()
                    wig.addItems(item.values)
                    wig.setCurrentText(item.value)
                    signal = wig.currentIndexChanged
                elif item.type == str and \
                        item.name in ('targetFile', 'maskerFile'):
                    wig = QtWidgets.QComboBox()
                    soundFiles = [''] + [sound.name for sound in gb.calibration]
                    soundFiles.sort()
                    wig.addItems(soundFiles)
                    wig.setMaximumWidth(200)
                    wig.setCurrentText(item.value)
                    signal = wig.currentIndexChanged
                elif item.type in (int, float, str) and not item.values:
                    wig = QtWidgets.QLineEdit(str(item.value))
                    signal = wig.textEdited
                    if item.name in ['targetFile', 'maskerFile']:
                        # make a small button for open file dialog
                        h = wig.sizeHint().height()
                        wig2 = guiHelper.makeFileButton('Select file')
                        wig2.clicked.connect(
                            functools.partial(self.paradigmButtonClicked, item))
                elif item.type == bool:
                    wig = QtWidgets.QCheckBox()
                    if item.value:
                        wig.setCheckState(QtCore.Qt.Checked)
                    else:
                        wig.setCheckState(QtCore.Qt.Unchecked)
                    signal = wig.stateChanged
                else:
                    raise ValueError('Unexpected parameter type')

                signal.connect(functools.partial(self.paradigmChanged, item))
                item.widget  = wig

                lbl = QtWidgets.QLabel(item.label + ':')
                layout2.addWidget(lbl, layout2.rowCount(), 0)

                if wig2 is None:
                    layout2.addWidget(wig, layout2.rowCount()-1, 1)
                else:
                    item.widget2 = wig2
                    itemLayout = QtWidgets.QHBoxLayout()
                    itemLayout.addWidget(wig)
                    itemLayout.addWidget(wig2)
                    layout2.addLayout(itemLayout, layout2.rowCount()-1, 1)

        frame = QtWidgets.QFrame()
        frame.setLayout(layout)

        scroll = QtWidgets.QScrollArea()
        scroll.setStyleSheet('.QScrollArea{border:%s}' % config.BORDER_STYLE)
        scroll.setWidgetResizable(True);
        scroll.setWidget(frame)
        scroll.setFixedWidth(scroll.sizeHint().width()*1.1)

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

    def initStatusBar(self):
        title = QtWidgets.QLabel('Current status')
        title.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtWidgets.QGridLayout()
        layout.setColumnStretch(0,1)
        layout.setColumnStretch(1,1)

        for item in gb.status:
            lbl = QtWidgets.QLabel(item.label + ':')
            txt = QtWidgets.QLineEdit(str(item.value))
            txt.setReadOnly(True)
            layout.addWidget(lbl, layout.rowCount(), 0)
            layout.addWidget(txt, layout.rowCount()-1, 1)
            item.widget = txt

        frame = QtWidgets.QFrame()
        frame.setLayout(layout)
        frame.setStyleSheet('.QFrame{border:%s}' % config.BORDER_STYLE)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(frame)

        frame = QtWidgets.QFrame()
        frame.setLayout(layout)

        bar = QtWidgets.QToolBar(title.text())
        bar.setMovable(False)
        bar.setStyleSheet('.QToolBar{border-left:%s;border-bottom:%s}' %
            (config.BORDER_STYLE, config.BORDER_STYLE))
        bar.addWidget(frame)

        return bar

    def initSoftControlBar(self):
        title = QtWidgets.QLabel('Software Control')
        title.setAlignment(QtCore.Qt.AlignCenter)

        layout1 = QtWidgets.QHBoxLayout()
        self.buttons.digitalInput = [None] * daqs.digitalInput.lineCount
        for lineID in range(daqs.digitalInput.lineCount):
            lineName = daqs.digitalInput.lineNames[lineID]
            btn = QtWidgets.QPushButton(lineName.title())
            btn.setCheckable(True)
            btn.setEnabled(False)
            btn.clicked.connect(self.digitalInputClicked)
            layout1.addWidget(btn)
            self.buttons.digitalInput[lineID] = btn

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout1)
        # layout.addLayout(layout2)

        frame = QtWidgets.QFrame()
        frame.setLayout(layout)
        frame.setStyleSheet('.QFrame{border:%s}' % config.BORDER_STYLE)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(frame)

        frame = QtWidgets.QFrame()
        frame.setLayout(layout)

        bar = QtWidgets.QToolBar(title.text())
        bar.setMovable(False)
        bar.setStyleSheet('.QToolBar{border-left:%s;border-bottom:%s}' %
            (config.BORDER_STYLE, config.BORDER_STYLE))
        bar.addWidget(frame)

        return bar

    def initTrialLogBar(self):
        title = QtWidgets.QLabel('Trial log')
        title.setAlignment(QtCore.Qt.AlignCenter)

        labels = []
        for item in gb.trial:
            if item.label:
                labels += [item.label]

        self.tblTrialLog = QtWidgets.QTableWidget(0, len(labels))
        self.tblTrialLog.setHorizontalHeaderLabels(labels)
        self.tblTrialLog.horizontalHeader().setResizeMode(
            QtWidgets.QHeaderView.Stretch)
        self.tblTrialLog.verticalHeader().setVisible(False)
        self.tblTrialLog.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        self.tblTrialLog.setFixedWidth(
            self.tblTrialLog.sizeHint().width()*1.4)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.tblTrialLog)

        frame = QtWidgets.QFrame()
        frame.setLayout(layout)

        bar = QtWidgets.QToolBar(title.text())
        bar.setMovable(False)
        bar.setStyleSheet('.QToolBar{border-left:%s}' % config.BORDER_STYLE)
        bar.addWidget(frame)

        return bar

    def initPerformanceBar(self):
        title = QtWidgets.QLabel('Performance')
        title.setAlignment(QtCore.Qt.AlignCenter)

        labels = []
        for item in gb.performance:
            if item.label:
                labels += [item.label]

        self.tblPerformance = QtWidgets.QTableWidget(0, len(labels))
        self.tblPerformance.setHorizontalHeaderLabels(labels)
        self.tblPerformance.horizontalHeader().setResizeMode(
            QtWidgets.QHeaderView.Stretch)
        self.tblPerformance.verticalHeader().setVisible(False)
        self.tblPerformance.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.tblPerformance)

        frame = QtWidgets.QFrame()
        frame.setLayout(layout)

        bar = QtWidgets.QToolBar(title.text())
        bar.setMovable(False)
        bar.setStyleSheet('.QToolBar{border-top:%s}' % config.BORDER_STYLE)
        bar.addWidget(frame)

        return bar

    ########################################
    # GUI callbacks

    @guiHelper.showExceptions
    def closeEvent(self, event):
        # if event.spontaneous()
        if gb.status.trialState.value not in gb.trialStatesAwaiting:
            res = guiHelper.showQuestion('Are you sure you want to quit?',
                'There is a trial in progress!')
        elif gb.status.experimentState.value == 'Running':
            res = guiHelper.showQuestion('Are you sure you want to quit?',
                'Experiment is still running!')
        else:
            res = True

        if not res:
            event.ignore()
            return

        self.pauseExperiment()
        self.pump.disconnect()
        daqs.clear()

    @guiHelper.showExceptions
    def keyPressEvent(self, event):
        # when an editable widget has focus do not act on key presses
        wig = QtWidgets.QApplication.focusWidget()
        if isinstance(wig, (QtWidgets.QLineEdit, QtWidgets.QComboBox,
                QtWidgets.QTableWidgetItem, QtWidgets.QCheckBox,
                QtWidgets.QTableWidget)) and not (
                hasattr(wig, 'isReadOnly') and wig.isReadOnly()):
            return

        # print('keyReleaseEvent: %d "%s" (focus: %s)' %
        #     (event.key(), event.text(), type(wig)))

        # start experiment with space or enter keys
        if (event.key() in (QtCore.Qt.Key_Space,
                QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and
                gb.status.experimentState.value in ('Not started', 'Paused')):
            self.startExperiment()
        # pause experiment with escape key
        elif (event.key() == QtCore.Qt.Key_Escape and
                gb.status.experimentState.value == 'Running'):
            self.pauseExperiment()

    @guiHelper.showExceptions
    def sessionChanged(self, item, *args):
        item.value = item.getWidgetValue()
        gb.session.overwriteData()

    @guiHelper.showExceptions
    def paradigmChanged(self, item, *args):
        # if item.name == 'targetType':
        #     tone = item.getWidgetValue() == 'Tone'
        #     if self.paradigm.targetFrequency.widget:
        #         self.paradigm.targetFrequency.widget.setEnabled(tone)
        #     if self.paradigm.targetFile.widget:
        #         self.paradigm.targetFile.widget .setEnabled(not tone)
        #         self.paradigm.targetFile.widget2.setEnabled(not tone)

        if gb.status.experimentState.value != 'Running':
            item.value = item.getWidgetValue()

        else:
            hasChanged = False
            for item in self.paradigm:
                if item.widget and item.value != item.getWidgetValue():
                    hasChanged = True
                    break

            self.buttons.apply .setEnabled(hasChanged)
            self.buttons.revert.setEnabled(hasChanged)

    @guiHelper.showExceptions
    def paradigmButtonClicked(self, item, *args):
        if item.name in ['targetFile', 'maskerFile']:
            file     = item.getWidgetValue()
            stimDir  = config.STIM_DIR
            if file:
                file = misc.absolutePath(file, stimDir)
            else:
                file = stimDir
            file     = guiHelper.openFile('Sound (*.wav)', file)

            if not file:
                return

            file = misc.relativePath(file, stimDir)

            item.widget.setText(file.replace('\\','/'))
            self.paradigmChanged(item)

    @guiHelper.showExceptions
    def roveAddClicked(self, *args):
        value = self.paradigm.rove.getWidgetValue()
        for name in value:
            value[name] += [str(self.paradigm[name].type())]
        self.paradigm.rove.setWidgetValue(value)
        self.paradigmChanged(self.paradigm.rove)

    @guiHelper.showExceptions
    def roveRemoveClicked(self, *args):
        wig = self.paradigm.rove.widget
        rows = wig.rowCount()
        r = wig.currentRow()
        if 2<=r and 3<rows:
            wig.removeRow(r)
            self.paradigmChanged(self.paradigm.rove)

    @guiHelper.showExceptions
    def applyClicked(self, *args):
        # keep a copy of the old paradigm
        paradigmOld = self.paradigm.copy()

        # transfer widget values to paradigm
        self.paradigm.applyWidgetValues()

        # evaluated right away if no trials ongoing
        if gb.status.trialState.value in gb.trialStatesAwaiting:
            res = self.evaluateParadigm()
        else:
            res = self.verifyParadigm()

        if res:
            self.buttons.apply .setEnabled(False)
            self.buttons.revert.setEnabled(False)
        else:
            paradigmOld.copyTo(self.paradigm)

    @guiHelper.showExceptions
    def revertClicked(self, *args):
        # transfer paradigm values to widgets
        self.paradigm.revertWidgetValues()

        self.buttons.apply .setEnabled(False)
        self.buttons.revert.setEnabled(False)

    @guiHelper.showExceptions
    def loadClicked(self, *args):
        file     = gb.session.paradigmFile.value
        dataDir  = gb.session.dataDir.value
        if file:
            file = misc.absolutePath(file, dataDir)
        else:
            file = dataDir
        file     = guiHelper.openFile(config.SETTINGS_FILTER, file)

        if not file:
            return

        # keep a copy of the old paradigm
        paradigmOld = self.paradigm.copy()

        # use a temp copy for loading new paradigm
        paradigmTemp = self.paradigm.copy()
        log.info('Loading paradigm from "%s"', file)
        paradigmTemp.loadFile(file)

        # verify rove parameters
        if self.paradigm.rove.keys != paradigmTemp.rove.keys:
            raise ValueError('Cannot load "%s"' % file,
                'Selected paradigm has different rove parameters. Try loading '
                'from initial setup window.')

        paradigmTemp.copyTo(self.paradigm)

        file = misc.relativePath(file, dataDir)
        gb.session.paradigmFile.value = file

        # transfer paradigm values to widgets
        self.paradigm.revertWidgetValues()

        if gb.status.experimentState == 'Running':
            # evaluate right away if no trials ongoing
            if gb.status.trialState.value in gb.trialStatesAwaiting:
                res = self.evaluateParadigm()
            else:
                res = self.verifyParadigm()

            if res:
                self.buttons.apply .setEnabled(False)
                self.buttons.revert.setEnabled(False)
            else:
                paradigmOld.copyTo(self.paradigm)

    @guiHelper.showExceptions
    def saveClicked(self, *args):
        file     = gb.session.paradigmFile.value
        dataDir  = gb.session.dataDir.value
        if file:
            file = misc.absolutePath(file, dataDir)
        else:
            file = dataDir
        file     = guiHelper.saveFile(config.SETTINGS_FILTER, file)

        if not file:
            return

        log.info('Saving paradigm to "%s"', file)
        self.paradigm.saveFile(file)

        file = misc.relativePath(file, dataDir)
        gb.session.paradigmFile.value = file

    @guiHelper.showExceptions
    def startClicked(self, *args):
        self.startExperiment()

    @guiHelper.showExceptions
    def pauseClicked(self, *args):
        if gb.status.trialState.value not in gb.trialStatesAwaiting:
            res = guiHelper.showQuestion('Are you sure you want to pause?',
                'There is a trial in progress!')
            if not res:
                return

        self.pauseExperiment()

    @guiHelper.showExceptions
    def goClicked(self, *args):
        log.info('Forcing trial type "Go remind"')
        self.forceTrialType = 'Go remind'

        if gb.status.experimentState == 'Running':
            # evaluate right away if no trials ongoing
            if gb.status.trialState.value in gb.trialStatesAwaiting:
                self.evaluateParadigm()
            else:
                self.verfyParadigm()
        else:
            gb.status.trialType.value = self.forceTrialType

    @guiHelper.showExceptions
    def nogoClicked(self, *args):
        log.info('Forcing trial type "Go remind"')
        self.forceTrialType = 'Nogo remind'

        if gb.status.experimentState == 'Running':
            # evaluate right away if no trials ongoing
            if gb.status.trialState.value in gb.trialStatesAwaiting:
                self.evaluateParadigm()
            else:
                self.verfyParadigm()
        else:
            gb.status.trialType.value = self.forceTrialType

    @guiHelper.showExceptions
    def trialClicked(self, *args):
        log.info('Triggering trial manually')
        if gb.status.trialState.value not in gb.trialStatesAwaiting:
            # raise RuntimeError(
            #     'Cannot start trial when another is in progress')
            return
        self.startTrial()

    @guiHelper.showExceptions
    def targetClicked(self, *args):
        log.info('Triggering target manually')
        self.triggerTarget()

    @guiHelper.showExceptions
    def pumpClicked(self, *args):
        log.info('Triggering pump manually')
        self.triggerPump()

    @guiHelper.showExceptions
    def timeoutClicked(self, *args):
        log.info('Triggering timeout manually')
        self.triggerTimeout()

    @guiHelper.showExceptions
    def exportClicked(self, *args):
        # example format: '36:15\t22\r\n\t30\r\n\t23\r\n\t40\r\n\t17'
        df        = gb.performance.dataFrame
        condCount = len(df)
        data      = ''

        # def get(name, fmt=None):
        #     if name in df:
        #         value = df[name][i]
        #     elif i==0:
        #         value = gb.trial.dataFrame[name].iloc[-1]
        #     else:
        #         return ''
        #
        #     if fmt:
        #         if np.isnan(value):
        #             return ''
        #         else:
        #             return fmt % value
        #     else:
        #         return value

        def get(name, fmt=None):
            res = '"'

            if name in df:
                for i in range(condCount):
                    value = df[name][i]
                    if callable(fmt): value = fmt(value)
                    elif fmt:         value = fmt % value
                    if value=='nan': value = ''
                    if i: res += '\r\n'
                    res += value

            elif len(df):
                value = gb.trial.dataFrame[name].iloc[-1]
                if callable(fmt): value = fmt(value)
                elif fmt:         value = fmt % value
                if value=='nan': value = ''
                res += value

            res += '"'
            return res

        # for i in range(condCount):
        i = 0
        # experiment start date
        if i==0: data += gb.experimentStart.strftime('%m/%d/%Y')
        data += '\t'

        # experiment start time
        if i==0: data += gb.experimentStart.strftime('%H:%M:%S')
        data += '\t'

        # experiment duration (mm:ss)
        if i==0: data += '%02d:%02d' % (
            gb.session.experimentDuration.value//60,
            gb.session.experimentDuration.value%60)
        data += '\t'

        # trial rate (trials per minute)
        if i==0: data += '%.3f' % (df.trialCount.sum() /
            gb.session.experimentDuration.value * 60)
        data += '\t'

        # total reward (ml)
        if i==0: data += '%g' % gb.session.totalReward.value
        data += '\t'

        data += get('rewardVolume'    , '%g'             ) + '\t'
        data += get('trialType'                          ) + '\t'
        data += get('trialCount'      , '%d'             ) + '\t'
        data += get('hitRate'         , '%.3f'           ) + '\t'
        data += get('faRate'          , '%.3f'           ) + '\t'
        data += get('targetFile'      , os.path.basename ) + '\t'
        data += get('targetLevel'     , '%g dB SPL'      ) + '\t'
        data += get('maskerFile'      , os.path.basename ) + '\t'
        data += get('maskerLevel'     , '%g dB SPL'      ) + '\t'
        # if self.maskerLevel is not None:
        #     data += '%g dB' % (df.targetLevel[i]-self.maskerLevel) + '\t'
        # else:
        #     data += '\t'
        data += get('pokeDuration'    , '%.3f'     ) + '\t'
        data += get('responseDuration', '%.3f'     ) + '\t'
        data += get('dPrime'          , '%.3f'     ) + '\t'

        # data += '\r\n'

        QtWidgets.QApplication.clipboard().setText(data)

    @guiHelper.showExceptions
    def closeClicked(self, *args):
        # if gb.status.experimentState.value == 'Running':
        #     res = guiHelper.showQuestion('Are you sure you want to quit?',
        #         'Experiment is still running!')
        #     if not res:
        #         return
        self.close()

    @guiHelper.showExceptions
    def digitalInputClicked(self, *args):
        data = [None] * len(self.buttons.digitalInput)
        for i in range(len(self.buttons.digitalInput)):
            data[i] = self.buttons.digitalInput[i].isChecked()
        daqs.digitalInput.write(data)

    @guiHelper.logExceptions
    def updateGUI(self):
        '''Called by `updateTimer` every 20ms.'''
        ts = daqs.getTS()
        gb.status.ts .value         = '%02d:%05.2f' % (ts//60, ts%60)
        gb.status.fps.value         = '%.1f' % self.plot.calculatedFPS
        gb.status.totalReward.value = '%.3f' % self.pump.getInfused()

    ########################################
    # DAQ callbacks

    def digitalInputEdgeDetected(self, task, name, edge, ts):
        log.debug('Detected %s edge on %s at %.3f', edge, name, ts)
        self.eventQueue.put(((name, edge), ts))

    def analogInputDataAcquired(self, task, data):
        speaker, mic, poke, spout = data
        self.speakerTrace.append(speaker)
        self.micTrace    .append(mic    )
        self.pokeTrace   .append(poke   )
        self.spoutTrace  .append(spout  )

    def analogOutputDataNeeded(self, task, nsWritten, nsNeeded):
        # random analog trace
        # t  = np.arange(nsWritten, nsWritten+nsNeeded) / task.fs
        # t += daqs.lastTS
        # x  = np.sin(t*2*np.pi) #+ np.random.randn(nsNeeded)/2
        # x += np.sin(t*2*np.pi*100)

        log.debug('%d samples needed at %d', nsNeeded, nsWritten)

        # prepare target
        if self.targetActive:
            targetData = self.getSound('target', nsWritten, nsNeeded)
        else:
            targetData = np.zeros(nsNeeded)

        # prepare masker
        maskerData = self.getSound('masker', nsWritten, nsNeeded)

        data = targetData + maskerData

        if config.SIM:
            noise = np.random.randn(nsNeeded)/5
            data  = np.array([data, data, noise, noise])

        return data
        # return np.random.randn(task.lineCount, nsNeeded)

    ########################################
    # event handling

    def startEventThread(self):
        # make sure the queue is empty before starting
        while not self.eventQueue:
            self.eventQueue.get()
        # start event processing
        self.eventThread = threading.Thread(target=self.eventLoop)
        self.eventThread.daemon = True
        self.eventThread.start()

    def stopEventThread(self):
        # stop event processing
        self.eventThreadStop.set()
        if self.eventThread:
            # use a loop to allow keyboard interrupt
            # while self.eventThread.isAlive():
            #     self.eventThread.join(.1)
            self.eventThread.join(1)
        self.stopEventTimer()
        self.eventThreadStop.clear()

    @guiHelper.logExceptions
    def eventLoop(self):
        log.debug('Starting event thread')

        while not self.eventThreadStop.wait(5e-3):
            # this lock prevents events from the event timer being added
            # to the queue while previous event is already being handled
            # which might end up stopping the event timer
            with self.eventLock:
                if self.eventQueue.empty(): continue

                (event, ts) = self.eventQueue.get()
                self.handleEvent(event, ts)

        log.debug('Stopping event thread')

    @guiHelper.logExceptions
    def handleEvent(self, event, ts):
        # prase events from `digitalInputEdgeDetected`
        if isinstance(event, tuple):
            if len(event) != 2:
                raise ValueError('`event` tuple should be exactly of length 2')

            name, edge = event
            epochs = {'poke':self.pokeEpoch, 'spout':self.spoutEpoch,
                'button':self.buttonEpoch}

            if name in epochs:
                if edge == 'rising':
                    epochs[name].start(ts)
                    event = name.title() + ' start'
                elif edge == 'falling':
                    epochs[name].stop(ts)
                    event = name.title() + ' stop'
                else:
                    raise ValueError('Invalid `edge` value "%s"' % edge)
            else:
                raise ValueError('Invalid line `name` "%s"' % name)

        mode  = gb.session.experimentMode.value
        state = gb.status.trialState.value

        log.info('Handling event "%s" in trial state "%s" at %.3f' %
            (event, state, ts))

        ##############################
        if mode == 'Passive':

            if state == 'Awaiting random trigger':
                if event == 'Random trigger':
                    self.startTrial()

            elif state == 'Trial ongoing':
                if event == 'Trial elapsed':
                    self.stopTrial(ts, 'None')

        ##############################
        elif mode == 'Spout training':

            if state == 'Awaiting spout':
                if event == 'Spout start':
                    gb.status.trialState.value = 'Spout active'
                    self.startTarget()
                    self.startPump()

            elif state == 'Spout active':
                if event == 'Spout stop':
                    gb.status.trialState.value = 'Awaiting spout'
                    self.stopTarget()
                    self.stopPump()
                    self.evaluateParadigm()

        ##############################
        elif mode == 'Target training':

            if state == 'Awaiting button':
                if event == 'Button start':
                    gb.status.trialState.value = 'Target active'
                    self.startTarget()
                elif event == 'Button stop':
                    raise RuntimeError('Should never happen!!')
                elif event == 'Spout start':
                    pass
                elif event == 'Spout stop':
                    self.stopPump()

            elif state == 'Target active':
                if event == 'Button start':
                    raise RuntimeError('Should never happen!!')
                if event == 'Button stop':
                    if self.pumpActive:
                        gb.status.trialState.value = 'Awaiting button'
                        self.stopTarget()
                        self.evaluateParadigm()
                    else:
                        gb.status.trialState.value = 'Response duration'
                        self.startEventTimer(
                            gb.trial.maxResponseDuration.value,
                            'Response duration elapsed')
                        self.stopTarget()
                elif event == 'Spout start':
                    self.startPump()
                elif event == 'Spout stop':
                    self.stopPump()

            elif state == 'Response duration':
                if event == 'Button start':
                    gb.status.trialState.value = 'Target active'
                    self.stopEventTimer()
                    self.startTarget()
                elif event == 'Button stop':
                    raise RuntimeError('Should never happen!!')
                elif event == 'Spout start':
                    gb.status.trialState.value = 'Awaiting button'
                    self.stopEventTimer()
                    self.startPump()
                    self.evaluateParadigm()
                elif event == 'Spout stop':
                    # can happen if spout was initiated in 'Awaiting button'
                    pass
                elif event == 'Response duration elapsed':
                    gb.status.trialState.value = 'Awaiting button'
                    self.evaluateParadigm()

        ##############################
        elif mode in ('Poke training', 'Go Nogo'):

            if state == 'Awaiting poke':
                if event == 'Poke start':
                    # animal has poked to initiate a trial
                    gb.status.trialState.value = 'Min poke duration'
                    self.startEventTimer(gb.trial.minPokeDuration.value,
                        'Min poke duration elapsed')
                    # if poke isn't maintained long enough, this value will
                    # get overwritten with the next poke
                    gb.trial.pokeStart.value = ts

            elif state == 'Min poke duration':
                if event == 'Poke stop':
                    # animal has withdrawn from poke too early
                    # cancel timer so it doesn't fire a
                    # 'Min poke duration elapsed'
                    log.info('Animal withdrew too early '
                        'during min poke period')
                    gb.status.trialState.value = 'Awaiting poke'
                    self.stopEventTimer()
                elif event == 'Min poke duration elapsed':
                    self.startTrial()

            elif state == 'Poke hold duration':
                if event == 'Poke stop':
                    log.info('Animal withdrew too early '
                        'during poke hold period')
                    log.info('Canceling trial at %.3f', ts)
                    gb.status.trialState.value = 'Intertrial duration'
                    self.stopEventTimer()
                    self.trialEpoch.stop(ts)
                    self.startEventTimer(gb.trial.intertrialDuration.value,
                        'Intertrial duration elapsed')
                    self.evaluateParadigm()
                elif event == 'Poke hold duration elapsed':
                    gb.status.trialState.value = 'Hold duration'
                    self.startEventTimer(gb.trial.holdDuration.value,
                        'Hold duration elapsed')

            elif state == 'Hold duration':
                # all animal-initiated events (poke/spout) are ignored
                # during this period but we may choose to record the time
                # of nose-poke withdraw if it occurs
                if event == 'Poke stop':
                    log.debug('Animal withdrew during hold period')
                    # record the time of nose-poke withdrawal if it is the
                    # first time since initiating a trial
                    if not gb.trial.pokeStop.value:
                        gb.trial.pokeStop.value = ts
                elif event == 'Hold duration elapsed':
                    gb.status.trialState.value = 'Response duration'
                    self.startEventTimer(gb.trial.maxResponseDuration.value,
                        'Response duration elapsed')

            elif state == 'Response duration':
                # if the animal happened to initiate a nose-poke during the
                # hold period above and is still maintaining the nose-poke,
                # they have to manually withdraw and re-poke for us to
                # process the event
                if event == 'Poke stop':
                    log.debug('Animal withdrew during response period')
                    # record the time of nose-poke withdrawal if it is the
                    # first time since initiating a trial.
                    if not gb.trial.pokeStop.value:
                        gb.trial.pokeStop.value = ts
                elif event == 'Poke start':    # repoke
                    self.stopEventTimer();
                    self.stopTrial(ts, 'Poke')
                elif event == 'Spout start':
                    self.stopEventTimer()
                    gb.trial.spoutStart.value = ts
                    self.stopTrial(ts, 'Spout')
                # elif event == 'Spout stop':
                #     pass
                elif event == 'Response duration elapsed':
                    self.stopTrial(ts, 'None')

            elif state == 'Timeout duration':
                if event == 'Timeout duration elapsed':
                    gb.status.trialState.value = 'Intertrial duration'
                    self.stopTimeout(recordTS=False)
                    self.startEventTimer(gb.trial.intertrialDuration.value,
                        'Intertrial duration elapsed')
                elif event in ('Spout start', 'Poke start'):
                    self.startEventTimer(gb.trial.timeoutDuration.value,
                        'Timeout duration elapsed')

            elif state == 'Intertrial duration':
                if event == 'Intertrial duration elapsed':
                    gb.status.trialState.value = 'Awaiting poke'

    def startEventTimer(self, duration, event):
        self.stopEventTimer()
        # this shouldn't necessarily be a threading Event, any mutable object
        # which can be passed by reference to the callback would do the trick
        self.eventTimerStop = threading.Event()
        self.eventTimer     = threading.Timer(duration,
            self.eventTimerCallback, [self.eventTimerStop, event])
        log.debug('Starting event timer for duration %.3f with event "%s"' %
            (duration, event))
        self.eventTimer.start()

    def stopEventTimer(self):
        if self.eventTimerStop:
            self.eventTimerStop.set()
        if self.eventTimer:
            self.eventTimer.cancel()

    @guiHelper.logExceptions
    def eventTimerCallback(self, eventTimerStop, event):
        # this lock prevents events from the event timer being added
        # to the queue while previous event is already being handled
        # which might end up stopping the event timer
        log.debug('Event timer callback with event "%s"' % event)
        with self.eventLock:
            if not eventTimerStop.is_set():
                log.debug('Putting event "%s" in queue' % event)
                self.eventQueue.put((event, daqs.getTS()))
            else:
                log.debug('Event "%s" was already stopped' % event)

    ########################################
    # experiment control

    @guiHelper.busyCursor
    def startExperiment(self):
        if not self.evaluateParadigm(): return

        if gb.status.experimentState.value == 'Not started':
            log.info('Starting experiment')
            if not gb.experimentStart:
                gb.experimentStart = dt.datetime.now()
                gb.session.experimentStart.value = gb.experimentStart.strftime(
                    config.DATETIME_FMT_MORE)
        elif gb.status.experimentState.value == 'Paused':
            log.info('Resuming experiment')

        # different experiment modes have different initial trial state
        expMode = gb.session.experimentMode.value
        if expMode == 'Passive':
            gb.status.trialState.value = 'Awaiting random trigger'
        elif expMode == 'Spout training':
            gb.status.trialState.value = 'Awaiting spout'
        elif expMode == 'Target training':
            gb.status.trialState.value = 'Awaiting button'
        elif expMode in ('Poke training', 'Go Nogo'):
            gb.status.trialState.value = 'Awaiting poke'
        else:
            raise ValueError('Invalid experiment mode "%s"' % expMode)

        self.startEventThread ()
        daqs            .start()
        self.plot       .start()
        self.updateTimer.start()

        gb.status.experimentState.value = 'Running'

        if gb.session.experimentMode.value == 'Passive':
            self.randomTrial()

        self.buttons.start  .setEnabled(False)
        self.buttons.pause  .setEnabled(True )
        self.buttons.export .setEnabled(False)
        self.buttons.trial  .setEnabled(True )
        self.buttons.target .setEnabled(True )
        self.buttons.pump   .setEnabled(True )
        self.buttons.timeout.setEnabled(True )
        if config.SIM:
            for btn in self.buttons.digitalInput:
                btn.setEnabled(True)

    @guiHelper.busyCursor
    def pauseExperiment(self):
        log.info('Pausing experiment')

        self.stopEventThread ()
        self.stopTarget      ()
        self.stopPump        ()
        self.stopTimeout     ()
        self.stopTrial       ()
        daqs            .stop()
        ts = daqs.getTS()
        self.pokeEpoch  .stop(ts)
        self.spoutEpoch .stop(ts)
        self.buttonEpoch.stop(ts)
        # TODO: zero pad analogInput, physiologyInput and probably
        # analogOutput._nsOffset to align them accroding to their fs
        ns         = daqs.analogOutput.nsGenerated
        nsInput    = int(ns / daqs.analogOutput.fs * daqs.analogInput.fs)
        nsAcquired = self.speakerTrace.ns
        nsNeeded   = nsInput - nsAcquired
        log.info('Zero-padding recordings (ns: %d, nsInput: %d, nsAcquired: '
            '%d, nsNeeded: %d)' % (ns, nsInput, nsAcquired, nsNeeded) )
        if nsNeeded>0:
            data = np.zeros((daqs.analogInput.lineCount, nsNeeded))
            self.analogInputDataAcquired(daqs.analogInput, data)
        self.plot       .stop()
        self.updateTimer.stop()

        # set to zero in order to output a ramp when resuming experiment
        self.maskerLevel                    = 0
        gb.status.experimentState.value     = 'Paused'
        gb.status.trialState.value          = 'None'

        gb.session.experimentStop.value     = dt.datetime.now().strftime(
            config.DATETIME_FMT_MORE)
        gb.session.experimentDuration.value = ts
        gb.session.totalReward.value        = self.pump.getInfused()
        gb.session.overwriteData()
        gb.session.saveFile(config.LAST_SESSION_FILE)
        log.info('Session context: %s', str(gb.session))

        hdf5.flush()

        self.buttons.start  .setEnabled(True )
        self.buttons.pause  .setEnabled(False)
        self.buttons.export .setEnabled(True )
        self.buttons.trial  .setEnabled(False)
        self.buttons.target .setEnabled(False)
        self.buttons.pump   .setEnabled(False)
        self.buttons.timeout.setEnabled(False)
        if config.SIM:
            for btn in self.buttons.digitalInput:
                btn.setEnabled(False)

    def startTrial(self):
        # if self.trialActive: return

        ts = daqs.getTS()
        log.info('Starting trial at %.3f', ts)

        if gb.session.experimentMode.value == 'Passive':
            gb.status.trialState.value = 'Trial ongoing'
            self.startEventTimer(gb.trial.targetDuration.value, 'Trial elapsed')

        elif gb.session.experimentMode.value in ('Poke training', 'Go Nogo'):
            if (gb.trial.trialType.value in gb.trialTypesNogo or
                    gb.trial.pokeHoldDuration.value == 0):
                # skip poke hold duration for nogo trials
                gb.status.trialState.value = 'Hold duration'
                self.startEventTimer(gb.trial.holdDuration.value,
                    'Hold duration elapsed')
            else:
                gb.status.trialState.value = 'Poke hold duration'
                self.startEventTimer(gb.trial.pokeHoldDuration.value,
                    'Poke hold duration elapsed')

        else:
            raise RuntimeError('Cannot `startTrial` in experiment mode "%s"' %
                gb.session.experimentMode.value)

        # start trial epoch
        self.trialEpoch.start(ts)
        gb.trial.trialStart.value = ts
        # self.trialActive = True

        self.triggerTarget()

        # reset forcing of next trial type
        self.forceTrialType = None

    def randomTrial(self):
        if gb.session.experimentMode.value != 'Passive':
            raise RuntimeError('No `randomTrial` in experiment mode "%s"' %
                gb.session.experimentMode.value)

        minDelay = (gb.trial.targetDuration.value +
            gb.trial.intertrialDuration.value)
        maxDelay = 10
        duration = np.random.rand()*(maxDelay-minDelay) + minDelay
        self.startEventTimer(duration, 'Random trigger')

    def stopTrial(self, ts=None, response=None):
        '''Stop the ongoing trial accroding to current experiment mode.

        Note: response='None' is a valid response in 'Go Nogo' experiment mode.
        However, respons`=None will only cancel the latest trial epoch (if any)
        and is mainly used when stopping the experiment abruptly.
        '''
        # nothing to stop if already in awaiting state
        # if gb.status.trialState.value in gb.trialStatesAwaiting:

        # nothing to stop if no trial active
        # if not self.trialActive:
        #     return

        if ts is None:
            ts = daqs.getTS()

        log.info('Stopping trial at %.3f', ts)

        self.trialEpoch.stop(ts)

        if response is None:
            return

        elif gb.session.experimentMode.value == 'Passive':
            gb.status.trialState.value = 'Awaiting random trigger'
            self.randomTrial()

        elif gb.session.experimentMode.value == 'Spout training':
            gb.status.trialState.value = 'Awaiting spout'
            self.stopEventTimer()
            self.stopTarget()
            self.stopPump()
            # no trial log or performance for spout training
            return

        elif gb.session.experimentMode.value == 'Tone training':
            gb.status.trialState.value = 'Awaiting button'
            self.stopEventTimer()
            self.stopTarget()
            self.stopPump()
            # no trial log or performance for spout training
            return

        elif gb.session.experimentMode.value in ('Poke training', 'Go Nogo'):
            if gb.trial.trialType.value in gb.trialTypesGo:
                score = 'HIT' if response == 'Spout' else 'MISS'
            elif gb.trial.trialType.value in gb.trialTypesNogo:
                score = 'FA'  if response == 'Spout' else 'CR'
            else:
                raise ValueError('Invalid trial type "%s"' %
                    gb.trial.trialType.value)

            log.info('Score is "%s"', score)

            if score == 'FA':
                self.triggerTimeout()
            else:
                if score == 'HIT':
                    self.triggerPump()

                log.debug('Entering intertrial duration')
                gb.status.trialState.value = 'Intertrial duration'
                self.startEventTimer(gb.trial.intertrialDuration.value,
                    'Intertrial duration elapsed')

            gb.trial.pokeDuration.value     = \
                gb.trial.pokeStop.value - gb.trial.pokeStart.value
            gb.trial.responseDuration.value = \
                np.nan if response == 'None'  \
                else ts - gb.trial.targetStart.value
            gb.trial.response.value         = response
            gb.trial.score.value            = score

        else:
            raise RuntimeError('Cannot `stopTrial` in experiment mode "%s"' %
                gb.session.experimentMode.value)

        # do after trial housekeeping
        self.updateTrialLog()
        gb.paradigm.appendData()
        gb.trial   .appendData()
        self.updatePerformance()
        log.info('Trial context: %s'      , str(gb.trial))
        performance = gb.performance.dataFrame.to_dict('list')
        log.info('Performance context: %s', str(performance))

        hdf5.flush()

        # evaluate paradigm to be ready for next trial
        self.evaluateParadigm()

    def getSound(self, name, ns, length):
        '''Get a cicular window into the specified sound data.

        Args:
            name (str): Should either be 'target' or 'masker'.
            ns (int): Number of sample to start the window from. This number
                will be substracted by the specified sound's `_StartNS`.
            length (int): The length of the circular window.
        '''
        if name not in ('target', 'masker'):
            raise ValueError('Invalid sound `name`: "%s"' % name)

        file    = getattr(self, name + 'File')
        level   = getattr(self, name + 'Level')
        startNS = getattr(self, name + 'StartNS')

        if not file or level==0 or np.isnan(level):
            return np.zeros(length)

        cal     = gb.calibration[os.path.basename(file)].value
        scale   = 10**((level-cal)/20)
        data    = self.soundData[file]

        window  = np.arange(ns-startNS, ns-startNS+length)
        window %= len(data)
        return data[window] * scale

    def getRamp(self, length):
        '''Get a sin^2 ramp with the specified length.'''
        return np.sin(np.pi/2*np.arange(length)/length)**2

    def getSimTTL(self, length):
        '''Deprecated'''
        fs = daqs.analogOutput.fs
        if not hasattr(self, '_lastState'): self._lastState = 0
        ramp = np.linspace(0, 1, 50e-3*fs)
        count = int(length / fs / .4)
        change = np.random.choice(length, count, False)
        change = np.sort(change)
        change = np.r_[0, change, length]
        y = np.zeros(length)
        for i in range(1, len(change)-1):
            if len(ramp) > length-change[i-1]: break
            lastState = (y[change[i-1]-1] if change[i-1]>0
                else self._lastState)
            newState = np.random.choice((0,5))
            y[change[i-1]:change[i-1]+len(ramp)] = \
                lastState*(1-ramp) + newState*ramp
            y[change[i-1]+len(ramp):change[i]] = newState
        self._lastState = y[-1]
        y += np.random.randn(length) / 5

        return y

    def updateTarget(self, mode):
        '''Triger, start or stop the target sound.

        Args:
            mode (str): Accepted values
        '''

        # if not self.targetFile: return

        if   mode=='start' and     self.targetActive: return
        elif mode=='stop'  and not self.targetActive: return

        # get the current position in the analog output buffer, and add a
        # certain `updateDelay` (to give us time to generate and upload the
        # new signal).
        fs  = daqs.analogOutput.fs
        ns  = daqs.analogOutput.nsGenerated
        ns += int(daqs.UPDATE_DELAY * fs)

        # the target at a specific phase of the modulated masker
        maskerFrequency = gb.trial.maskerFrequency.value
        if (mode=='trigger' and maskerFrequency != 0 and
                not np.isnan(maskerFrequency)):
            periodNS       = fs / maskerFrequency
            phaseDelayNS   = gb.trial.phaseDelay.value/360 * periodNS
            currentPhaseNS = (ns - self.maskerStartNS) % periodNS
            delayNS        = phaseDelayNS - currentPhaseNS
            if delayNS < 0:
                delayNS   += periodNS
            ns            += int(delayNS)

        if mode=='trigger':
            targetDuration     = gb.trial.targetDuration.value
            if targetDuration==0 or np.isnan(targetDuration):
                targetDuration = len(self.soundData[self.targetFile])/fs
            self.targetStartNS = ns
        elif mode=='start':
            targetDuration     = daqs.analogOutput.dataChunk
            self.targetStartNS = ns
        elif mode=='stop':
            targetDuration     = gb.trial.targetRamp.value * 1e-3
        else:
            raise ValueError('Wrong value given for the mode parameter. '
                'Should be "trigger", "start" or "stop"')

        targetLength  = int(targetDuration * fs)
        targetData    = self.getSound('target', ns, targetLength)

        # ramp beginning and end of the target
        targetRampLength = int(gb.trial.targetRamp.value * 1e-3 * fs)
        if targetRampLength != 0:
            targetRamp = self.getRamp(targetRampLength)
            if mode=='trigger' or mode=='start':
                targetData[0:targetRampLength] *= targetRamp
            if mode=='trigger' or mode=='stop':
                targetData[-targetRampLength:] *= targetRamp[::-1]

        # zero-pad if target is less than `minLength`
        minLength = int(daqs.analogOutput.dataChunk * fs)
        if targetLength < minLength:
            targetData = np.concatenate(
                [targetData, np.zeros(minLength-targetLength)])
            targetLength = len(targetData)

        # combine target with target and masker
        maskerData = self.getSound('masker', ns, targetLength)
        data       = targetData + maskerData

        if config.SIM:
            noise = np.random.randn(targetLength)/5
            data  = np.array([data, data, noise, noise])

        # update here before analogOutput.write to be ready for an
        # immediate call to analogOutput.dataNeeded
        self.targetActive = mode=='start'

        log.debug('Updating analog output buffer at %.3f for %.3f duration' %
            (ns/fs, targetLength/fs))
        daqs.analogOutput.write(data, ns)

        log.debug('Logging target epoch to HDF5')
        file = os.path.basename(self.targetFile)
        ts = ns / fs
        if mode=='trigger':
            self.targetEpoch.append(ts, ts+targetDuration)
            gb.trial.targetStart.value = ts
            gb.trial.targetStop.value  = ts+targetDuration
            log.info('Triggering target "%s" at %.3f for %g duration',
                file, ts, targetDuration)
        elif mode=='start':
            self.targetEpoch.start(ts)
            gb.trial.targetStart.value = ts
            log.info('Starting target "%s" at %.3f', file, ts)
        elif mode=='stop':
            self.targetEpoch.stop(ts+targetDuration)
            gb.trial.targetStop.value = ts+targetDuration
            log.info('Stopping target "%s" at %.3f', file, ts)

    def startTarget(self):
        self.updateTarget('start')

    def stopTarget(self):
        self.updateTarget('stop')

    def triggerTarget(self):
        self.updateTarget('trigger')

    def updateMasker(self):
        # do nothing if neither masker file nor masker level have changed
        if (self.maskerFile == gb.trial.maskerFile.value and
                self.maskerLevel == gb.trial.maskerLevel.value):
            return

        fs = daqs.analogOutput.fs
        # where to insert the updated masker
        ns = daqs.analogOutput.nsGenerated
        if gb.status.experimentState.value == 'Running':
            ns += int(daqs.UPDATE_DELAY * fs)

        dataLength     = int(daqs.analogOutput.dataChunk * fs)
        rampLength     = int(.5 * fs)
        ramp           = self.getRamp(rampLength)

        # ramp down old masker and zero-pad
        oldMaskerData  = self.getSound('masker', ns, rampLength)
        oldMaskerData *= ramp[::-1]
        oldMaskerData  = np.r_[oldMaskerData, np.zeros(dataLength-rampLength)]

        # save starting sample of the new masker
        if self.maskerFile != gb.trial.maskerFile.value:
            self.maskerStartNS = ns
        # update local copies of masker file and level
        self.maskerFile  = gb.trial.maskerFile .value
        self.maskerLevel = gb.trial.maskerLevel.value

        # ramp up new masker
        newMaskerData               = self.getSound('masker', ns, dataLength)
        newMaskerData[0:rampLength] *= ramp

        # also load target if it happens to be active
        if self.targetActive:
            targetData = self.getSound('target', ns, dataLength)
        else:
            targetData = np.zeros(dataLength)

        # mix old masker, new masker and target sounds
        data = oldMaskerData + newMaskerData + targetData

        if config.SIM:
            noise = np.random.randn(dataLength)/5
            data  = np.array([data, data, noise, noise])

        log.debug('Updating analog output buffer at %.3f for %.3f duration' %
            (ns/fs, dataLength/fs))
        daqs.analogOutput.write(data, ns)

    def updatePump(self):
        if self.pumpActive:
            self.pumpUpdateReq = True
            log.info('Requesting pump update')
            return
        self.pumpUpdateReq = False

        log.info('Updating pump')
        self.pump.setRate    (gb.trial.pumpRate    .value       )    # ml/min
        self.pump.setVolume  (gb.trial.rewardVolume.value * 1e-3)    # ml
        syringeDiameter = gb.syringeDiameters[gb.trial.syringeType.value]
        # if self.syringeDiameter != syringeDiameter:
        #     self.syringeDiameter = syringeDiameter
        self.pump.setDiameter(syringeDiameter)
        # self.pump.resume()

    def startPump(self):
        if self.pumpActive: return

        log.info('Starting pump')
        # self.pump.pause()
        self.pump.setVolume(0)
        # self.pump.resume()
        self.pump.start()
        # record timestamp
        ts = daqs.getTS()
        self.pumpEpoch.start(ts)
        gb.trial.pumpStart.value = ts
        self.pumpActive = True

    def stopPump(self):
        if not self.pumpActive: return

        log.info('Stopping pump')
        self.pump.stop()
        # self.pump.pause()
        self.pump.setVolume(gb.trial.rewardVolume.value * 1e-3)    # ml
        # self.pump.resume()
        # record timestamp
        ts = daqs.getTS()
        self.pumpEpoch.stop(ts)
        gb.trial.pumpStop.value = ts
        self.pumpActive = False
        if self.pumpUpdateReq:
            self.updatePump()

    def triggerPump(self):
        if self.pumpActive: return
        if gb.trial.rewardVolume.value < 0.5: return

        log.info('Triggering pump')
        self.pump.start()
        # record timestamp
        ts = daqs.getTS()
        duration = gb.trial.rewardVolume.value*60/gb.trial.pumpRate.value/1000
        self.pumpEpoch.append(ts, ts+duration)
        gb.trial.pumpStart.value = ts
        gb.trial.pumpStop .value = ts+duration

        self.pumpActive = True
        timer = threading.Timer(duration+50e-3, self.triggerPumpCallback)
        timer.start()

    def triggerPumpCallback(self):
        self.pumpActive = False
        if self.pumpUpdateReq:
            self.updatePump()

    def startTimeout(self, recordTS=True):
        log.info('Starting timeout')
        daqs.digitalOutput.write(0)
        self.timeoutActive = True
        # record timestamp
        if recordTS:
            ts = daqs.getTS()
            self.timeoutEpoch.start(ts)
            gb.trial.timeoutStart.value = ts

    def stopTimeout(self, recordTS=True):
        if not self.timeoutActive: return
        log.info('Stopping timeout')
        daqs.digitalOutput.write(1)
        self.timeoutActive = False
        # record timestamp
        if recordTS:
            ts = daqs.getTS()
            self.timeoutEpoch.stop(ts)
            gb.trial.timeoutStop.value = ts

    def triggerTimeout(self):
        if gb.session.experimentMode.value not in ('Poke training', 'Go Nogo'):
            log.warning('Cannot trigger timeout in experiment mode "%s"' %
                gb.session.experimentMode.value)
            return
        self.startTimeout(False)
        # record timestamp
        ts = daqs.getTS()
        duration = gb.trial.timeoutDuration.value
        self.timeoutEpoch.append(ts, ts+duration)
        gb.trial.timeoutStart.value = ts
        gb.trial.timeoutStop.value  = ts+duration
        # setup timer to stop timeout
        gb.status.trialState.value = 'Timeout duration'
        self.startEventTimer(duration, 'Timeout duration elapsed')

    ########################################
    # house keeping

    def runInGUIThread(self, func, blocking=True, args=[], kwargs={}):
        event = threading.Event()
        self.runInGUIThreadSignal.emit(func, event, args, kwargs)
        if blocking: event.wait()

    def runInGUIThreadCallback(self, func, event, args, kwargs):
        try:
            func(*args, **kwargs)
        except:
            log.exception('')
            guiHelper.showException()
        finally:
            event.set()

    def roveSetWidgetValue(self, item, value):
        params = list(value.keys())
        values = list(value.values())
        cols   = len(params)
        rows   = len(values[0])

        def colorMap(index):
            if   index == 0: return config.COLOR_NOGO
            elif index == 1: return config.COLOR_GOREMIND
            else:            return config.COLOR_GO

        item.widget.setRowCount(rows)
        item.widget.blockSignals(True)
        for c in range(cols):
            for r in range(rows):
                if params[c] in ('targetFile', 'maskerFile'):
                    cell = QtWidgets.QComboBox()
                    soundFiles = [''] + [sound.name for sound in gb.calibration]
                    soundFiles.sort()
                    cell.addItems(soundFiles)
                    style = ('QComboBox{background-color: rgb(%d,%d,%d); }' %
                        colorMap(r) )
                    cell.setStyleSheet(style)
                    cell.setCurrentText(values[c][r])
                    cell.currentIndexChanged.connect(functools.partial(
                        self.paradigmChanged, item))
                    item.widget.setCellWidget(r, c, cell)
                else:
                    cell = QtWidgets.QTableWidgetItem()
                    cell.setText(values[c][r])
                    # cell.setTextAlignment(QtCore.Qt.AlignCenter)
                    cell.setBackground(QtGui.QBrush(QtGui.QColor(*colorMap(r))))
                    item.widget.setItem(r, c, cell)
        item.widget.blockSignals(False)

    def roveGetWidgetValue(self, item):
        rows  = item.widget.rowCount()
        cols  = item.widget.columnCount()
        keys  = list(item.value.keys())
        value = {}

        for c in range(cols):
            col = []
            for r in range(rows):
                cell = item.widget.item(r,c)
                if not cell:
                    cell = item.widget.cellWidget(r,c)
                if isinstance(cell, QtWidgets.QComboBox):
                    col += [cell.currentText()]
                elif isinstance(cell, QtWidgets.QTableWidgetItem):
                    col += [cell.text()]
                else:
                    raise ValueError('Invalid rove cell "%s" at %dx%d' %
                        (type(cell), c, r))
            value[keys[c]] = col

        return value

    def evaluateItem(self, itemName, itemType, itemValue, locals=None):
        shortLabel = self.paradigm[itemName].shortLabel
        # evaluation needed
        if itemType != str and isinstance(itemValue, str):
            # evaluate if string not empty
            if itemValue:
                try:
                    evalValue = itemType(eval(itemValue, None, locals))
                except Exception as e:
                    raise ValueError('Cannot evaluate "%s"' % shortLabel) from e
            else:
                raise ValueError('"%s" cannot be empty' % shortLabel)
        # item type is string, no evaluation needed
        else:
            evalValue = itemType(itemValue)

        # load sound file
        if itemName in ('targetFile', 'maskerFile') and evalValue and \
                evalValue not in self.soundData:
            try:
                log.info('Loading "%s": "%s"', shortLabel, evalValue)
                file = misc.absolutePath(evalValue, config.STIM_DIR)
                fs, data = sp.io.wavfile.read(file, mmap=True)
            except Exception as e:
                raise ValueError('Cannot load "%s": "%s"' %
                    (shortLabel, evalValue)) from e

            # check if file has calibration
            if os.path.basename(evalValue) not in gb.calibration:
                raise ValueError('Specified "%s": "%s" has no calibration' %
                    (shortLabel, evalValue))
            # check for sampling frequency
            if fs != daqs.analogOutput.fs:
                raise ValueError('Sampling frequency of "%s": "%s" should be %g'
                    % (shortLabel, evalValue, daqs.analogOutput.fs))
            # check if file is empty
            if not len(data):
                raise ValueError('Specified "%s": "%s" is empty' %
                    (shortLabel, evalValue))

            # substract dc and normalize by rms
            data = data.astype('float64')
            data -= data.mean()
            data *= 1/np.sqrt((data**2).mean())

            # store data
            self.soundData[evalValue] = data

        return copy.deepcopy(evalValue)

    def evaluateParadigm(self, verifyOnly=False):
        try:
            if verifyOnly:
                log.info('Verifying paradigm')
            else:
                 log.info('Evaluating paradigm')

            self.trial.clearValues()

            # count number of last consecutive nogo trials
            nogoCount = 0
            for trialType in reversed(gb.trial.dataFrame.trialType):
                if trialType in gb.trialTypesNogo:
                    nogoCount += 1
                else:
                    break

            # fetch last trial's score
            lastScore = None
            if len(gb.trial.dataFrame):
                lastScore = gb.trial.dataFrame.score.iloc[-1]

            # evaluate paradigm and store the results in trial context
            for item in self.paradigm:
                if item.type and item.name not in self.paradigm.rove.value:
                    self.trial[item.name].value = self.evaluateItem(
                        item.name, item.type, item.value, locals())

            # evaluate rove parameters
            for name, value in self.trial.rove.value.items():
                itemType = self.trial[name].type
                shortLabel = self.paradigm[name].shortLabel
                for i in range(len(value)):
                    value[i] = self.evaluateItem(name, itemType, value[i],
                        locals())

            # determine next trial type (go/nogo)
            if self.forceTrialType:
                self.trial.trialType.value = self.forceTrialType
            elif (self.trial.repeatFA.value and lastScore=='FA'):
                self.trial.trialType.value = 'Nogo repeat'
            elif self.trial.goProbability.value <= np.random.rand():
                self.trial.trialType.value = 'Nogo'
            else:
                self.trial.trialType.value = 'Go'
            gb.status.trialType.value = self.trial.trialType.value

            if not verifyOnly:
                log.info('Setting trial type to "%s"',
                    self.trial.trialType.value)

            # select appropriate rove parameters

            if self.trial.goOrder.value != 'Random':
                raise NotImplementedError('Selected "Go order": "%s" is not '
                    'implemented yet' % self.trial.goOrder.value)
            elif self.trial.trialType.value in gb.trialTypesNogo:
                roveIndex = 0
            elif self.trial.trialType.value == 'Go remind':
                roveIndex = 1
            elif self.trial.goOrder.value == 'Random':
                roveConditions = len(list(self.trial.rove.value.values())[0])
                roveIndex = int(np.random.rand()*(roveConditions-2)+2)

            # apply rove parameters to trial context
            for name, value in self.trial.rove.value.items():
                self.trial[name].value = value[roveIndex]
                # also keep a comma separated string of evaluated rove values
                # if self.trial.roveValues.value:
                #     self.trial.roveValues.value += ', '
                # self.trial.roveValues.value += str(value[roveIndex])

            # check target+masker voltage range
            if    ( 'targetFile'  in self.trial.rove.value or
                    'targetLevel' in self.trial.rove.value or
                    'maskerFile'  in self.trial.rove.value or
                    'maskerLevel' in self.trial.rove.value ):
                conds = len(list(self.trial.rove.value.values())[0])
            else:
                conds = 1

            def getParam(name):
                if name in self.trial.rove.value:
                    return self.trial.rove.value[name][i]
                else:
                    return self.trial[name].value

            for i in range(conds):
                minVoltage = 0
                maxVoltage = 0

                targetFile  = getParam('targetFile' )
                targetLevel = getParam('targetLevel')
                if targetFile and targetLevel!=0 and not np.isnan(targetLevel):
                    cal   = gb.calibration[os.path.basename(targetFile)].value
                    scale = 10**((targetLevel-cal)/20)
                    data  = self.soundData[targetFile]
                    minVoltage += data.min() * scale
                    maxVoltage += data.max() * scale

                maskerFile  = getParam('maskerFile' )
                maskerLevel = getParam('maskerLevel')
                if maskerFile and maskerLevel!=0 and not np.isnan(maskerLevel):
                    cal   = gb.calibration[os.path.basename(maskerFile)].value
                    scale = 10**((maskerLevel-cal)/20)
                    data  = self.soundData[maskerFile]
                    minVoltage += data.min() * scale
                    maxVoltage += data.max() * scale

                if (minVoltage < daqs.MIN_VOLTAGE or
                        daqs.MAX_VOLTAGE < maxVoltage):
                    raise ValueError('Cannot exceed voltage range (%g, %g) of '
                        'the DAQ (Target file: %s, target level: %g, masker '
                        'file: %s, masker level: %g)' % (daqs.MIN_VOLTAGE,
                        daqs.MAX_VOLTAGE, targetFile, targetLevel, maskerFile,
                        maskerLevel))

            # update current paradigm and trial settings
            if not verifyOnly:
                self.paradigm.copyTo(gb.paradigm)
                self.trial   .copyTo(gb.trial   )

                self.targetFile  = self.trial.targetFile.value
                self.targetLevel = self.trial.targetLevel.value

                # if not hasattr(self, 'xy'): self.xy = 0
                # self.xy += 1
                # if self.xy == 3: raise SystemError('>:D')

                self.updateMasker()
                self.updatePump()

            return True

        except Exception as e:
            log.exception('')
            if threading.current_thread() == self.guiThread:
                guiHelper.showException()
            else:
                self.runInGUIThread(guiHelper.showException, False, [e])

            return False

    def verifyParadigm(self):
        return self.evaluateParadigm(verifyOnly=True)

    def updateTrialLog(self):
        # make sure function is only executed in the GUI thread
        if threading.current_thread() != self.guiThread:
            self.runInGUIThread(self.updateTrialLog)
            return

        self.tblTrialLog.insertRow(0)
        brush = QtGui.QBrush(QtGui.QColor(
            *config.COLOR_MAP[gb.trial.trialType.value]))
        col = 0

        for item in gb.trial:
            if item.label:
                if item.type == float:
                    # show user-entered float values as they are
                    if item.name in gb.paradigm.rove.value:
                        text = '%g' % item.value
                    # show calculated float values with 2 decimal points
                    else:
                        text = '%.2f' % item.value
                else:
                    text = str(item.value)
                # instead of nans show an empty cell
                if text == 'nan':
                    text = ''
                # show only file name
                if item.name in ('targetFile', 'maskerFile'):
                    text = os.path.basename(text)
                # if item.name=='roveValues': text = text.replace(', ', '\n')

                cell = QtWidgets.QTableWidgetItem()
                cell.setText(text)
                cell.setTextAlignment(QtCore.Qt.AlignCenter)
                cell.setBackground(brush)
                cell.setFlags(QtCore.Qt.ItemIsEnabled)
                self.tblTrialLog.setItem(0, col, cell)

                col += 1

    def updatePerformance(self):
        # make sure function is only executed in the GUI thread
        if threading.current_thread() != self.guiThread:
            self.runInGUIThread(self.updatePerformance)
            return

        trials = gb.trial.dataFrame.copy()

        if len(trials)==0: return

        # needed for grouping
        trials.loc[trials.trialType=='Go remind'  , 'trialType'] = 'Go'
        trials.loc[trials.trialType=='Nogo remind', 'trialType'] = 'Nogo'
        trials.loc[trials.trialType=='Nogo repeat', 'trialType'] = 'Nogo'

        # rove params are used in the grouping to aid sorting
        roveParams = list(gb.trial.rove.value.keys())
        groupParams = roveParams + ['trialType']
        # count number of trials within each group
        df = trials.groupby(groupParams + ['score']).size()
        # transfer each score into its own column
        df = df.unstack('score').rename_axis('', axis='columns')
        # add missing scores in columns and fill them with 0
        df = df.reindex(gb.trial.score.values, axis='columns').fillna(0)
        # rename columns: HIT -> hitCount
        df = df.rename(lambda x: x.lower() + 'Count', axis='columns')

        df['trialCount'] = df.sum(axis=1)
        df['hitRate'   ] = df.hitCount/(df.hitCount+df.missCount)
        df['faRate'    ] = df.faCount /(df.crCount +df.faCount  )

        # compute median poke and trial duration
        median = trials.groupby(groupParams) \
            [['pokeDuration', 'responseDuration']].median()
        df = df.join(median)

        # transfer rove params and trialType from index to individual columns
        df = df.reset_index()
        # nogo row(s) should be first
        df = df.sort_values(by='trialType', ascending=False)
        df = df.reset_index(drop=True)

        df.loc[df.trialType=='Go'  , ['crCount' , 'faCount'  ]] = np.nan
        df.loc[df.trialType=='Nogo', ['hitCount', 'missCount']] = np.nan

        # clip hit and FA rates to 0.05 and 0.95 and calculate d' sensitivity by
        # substracting PPF of hit rate of each go condition from PPF of FA rate
        # of the sole nogo condition. Nogo is assumed to be the first condition
        # in the list and always has a d' of zero
        # PPF: percent point function
        ppfFunc = lambda x: np.nan if np.isnan(x) else sp.stats.norm.ppf(x)
        ppf = df[['hitRate', 'faRate']].clip(0.05, 0.95).applymap(ppfFunc)
        df['dPrime'] = ppf.hitRate - ppf.faRate.values[0]
        df.dPrime.values[0] = np.nan

        # save calculated performance in HDF5 file and show them in the table
        gb.performance.clearData()
        self.tblPerformance.setRowCount(len(df))

        for row in range(len(df)):
            brush = QtGui.QBrush(QtGui.QColor(
                *config.COLOR_MAP[df.trialType[row]]))
            col = 0
            gb.performance.clearValues()

            for item in gb.performance:
                item.value = df[item.name][row]

                if item.label:
                    if item.type == float:
                        # show user-entered float values as they are
                        if item.name in gb.paradigm.rove.value:
                            text = '%g' % item.value
                        # show counts as ints
                        # note: count types are specified as float to allow
                        # setting to np.nan
                        elif item.name in ('trialCount', 'missCount',
                                'crCount', 'faCount'):
                            text = '%d' % item.value
                        # show calculated float values with 3 decimal points
                        else:
                            text = '%.3f' % item.value
                    elif item.type == int:
                        text = '%d' % item.value
                    else:
                        text = str(item.value)
                    # instead of nans show an empty cell
                    if text == 'nan':
                        text = ''
                    # show only file name
                    if item.name in ('targetFile', 'maskerFile'):
                        text = os.path.basename(text)

                    # create and set the perfromance table cell
                    cell = QtWidgets.QTableWidgetItem()
                    cell.setText(text)
                    cell.setTextAlignment(QtCore.Qt.AlignCenter)
                    cell.setBackground(brush)
                    cell.setFlags(QtCore.Qt.ItemIsEnabled)
                    self.tblPerformance.setItem(row, col, cell)

                    # advance to next column only if the current item has a
                    # label, otherwise the item is not be shown in the table
                    col += 1

            # write the current row of performance data to HDF5 file
            gb.performance.appendData()

# playground
if __name__ == '__main__':
    pass


# end
