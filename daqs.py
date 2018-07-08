'''Setup DAQ tasks for the experiment, either in behavior or physiology modes.

This module makes the DAQ tasks accessible from all other modules.


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
import numpy   as np

import daq
import config
import globals as gb


log = logging.getLogger(__name__)

UPDATE_DELAY     = 50e-3    # seconds
MAX_VOLTAGE      = +10
MIN_VOLTAGE      = -10

digitalInput     = None
digitalOutput    = None
analogInput      = None
physiologyInput  = None
analogOutput     = None
physiologyOutput = None

def init():
    global digitalInput, digitalOutput, analogInput, \
        physiologyInput, analogOutput, physiologyOutput

    devA = 'dev1'
    devB = 'dev2'
    physLines = 15

    if gb.appMode != 'Calibration':
        digitalInput = daq.DigitalInput('/%s/port0/line1:3' % devA,
            name='digitalInput', getTS=getTS,
            lineNames=['spout', 'poke', 'button'])

        digitalOutput = daq.DigitalOutput('/%s/port1/line1' % devA,
            name='digitalOutput', initialState=1)
        digitalOutput.start()
        digitalOutput.write(1)    # turn the light on
        digitalOutput.stop()

        if gb.session.recording.value == 'Physiology':
            # slave task in sync with `analogOutput`
            physiologyInput = daq.AnalogInput('/%s/ai1:%d' % (devB, physLines),
                31.25e3, np.inf, name='physiologyInput', dataChunk=20e-3)
                # timebaseSrc='/%s/20MHzTimebase' % devA, timebaseRate=20e6,
                # startTrigger='/%s/ao/StartTrigger' % devA)

            # slave task in sync with `analogOutput`
        analogInput = daq.AnalogInput('/%s/ai0:3' % devA, 20e3, np.inf,
            name='analogInput', dataChunk=10e-3,
            timebaseSrc='/%s/20MHzTimebase' % devA,
            startTrigger='/%s/ao/StartTrigger' % devA)

    # master task
    if config.SIM and gb.appMode != 'Calibration':
        lines = '/%s/ao0:3' % devA
    else:
        lines = '/%s/ao0' % devA
    analogOutput = daq.AnalogOutput(lines, 100e3, np.inf,
        name='analogOutput', dataChunk=2,
        timebaseSrc='/%s/20MHzTimebase' % devA)

    if config.SIM and gb.appMode != 'Calibration':
        analogOutput.connect(analogInput, [0,1,2,3])
        if gb.session.recording.value == 'Physiology':
            physiologyOutput = daq.AnalogOutput('/%s/ao1:%d' % (devB, physLines),
                31.25e3, np.inf, name='physiologyOutput', dataChunk=2,
                dataNeeded=physiologyOutputDataNeeded)
            physiologyOutput.connect(physiologyInput, list(range(physLines)))

def start():
    if gb.appMode != 'Calibration':
        digitalInput.start()
        digitalOutput.start()
        if gb.session.recording.value == 'Physiology':
            physiologyInput.start()
        analogInput.start()
    analogOutput.start()    # all analog tasks actually start here
    if config.SIM and gb.appMode != 'Calibration' and \
            gb.session.recording.value == 'Physiology':
        physiologyOutput.start()

def stop():
    if gb.appMode != 'Calibration':
        digitalInput.stop()
        digitalOutput.write(1)    # leave the light on
        digitalOutput.stop()
        if gb.session.recording.value == 'Physiology':
            physiologyInput.stop()
        analogInput.stop()
    analogOutput.stop()
    if config.SIM and gb.appMode != 'Calibration' and \
            gb.session.recording.value == 'Physiology':
        physiologyOutput.stop()

def clear():
    if gb.appMode != 'Calibration':
        digitalInput.clear()
        digitalOutput.clear()
        if gb.session.recording.value == 'Physiology':
            physiologyInput.clear()
        analogInput.clear()
    analogOutput.clear()
    if config.SIM and gb.appMode != 'Calibration' and \
            gb.session.recording.value == 'Physiology':
        physiologyOutput.clear()

def getTS():
    return analogOutput.nsGenerated / analogOutput.fs

def physiologyOutputDataNeeded(task, nsWritten, nsNeeded):
    data = np.random.randn(task.lineCount, nsNeeded)
    return data
