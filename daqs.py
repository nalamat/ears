'''Setup DAQ tasks for the experiment, either in behavior or physiology modes.

This module allows central access to DAQ task instances from all other modules.


This file is part of the EARS project <https://github.com/nalamat/ears>
Copyright (C) 2017-2021 Nima Alamatsaz <nima.alamatsaz@gmail.com>
'''

import logging
import numpy     as np

import config
import globals   as gb
import EasyDAQmx as daq


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

    if gb.appMode != 'Calibration':
        digitalInput = daq.DigitalInput('/%s/port0/line1:3' % config.DAQ_A,
            name='digitalInput', getTS=getTS,
            lineNames=['poke', 'spout', 'button'])

        digitalOutput = daq.DigitalOutput('/%s/port1/line1' % config.DAQ_A,
            name='digitalOutput', initialState=1)
        digitalOutput.start()
        digitalOutput.write(1)    # turn the light on
        digitalOutput.stop()

        if gb.session.recording.value == 'Physiology':
            # slave task in sync with `analogOutput`
            physiologyInput = daq.AnalogInput('/%s/ai1:%d' % (config.DAQ_B,
                config.ELECTRODE_COUNT), 31.25e3, np.inf,
                name='physiologyInput', dataChunk=20e-3, timebaseRate=20e6,
                timebaseSrc='/%s/20MHzTimebase' % config.DAQ_A,
                startTrigger='/%s/ao/StartTrigger' % config.DAQ_A)

            # slave task in sync with `analogOutput`
        analogInput = daq.AnalogInput('/%s/ai0:3' % config.DAQ_A, 20e3, np.inf,
            name='analogInput', dataChunk=10e-3,
            timebaseSrc='/%s/20MHzTimebase' % config.DAQ_A,
            startTrigger='/%s/ao/StartTrigger' % config.DAQ_A)

    # master task
    if config.SIM and gb.appMode != 'Calibration':
        lines = '/%s/ao0:3' % config.DAQ_A
    else:
        lines = '/%s/ao0' % config.DAQ_A
    analogOutput = daq.AnalogOutput(lines, 100e3, np.inf,
        name='analogOutput', dataChunk=2,
        timebaseSrc='/%s/20MHzTimebase' % config.DAQ_A)

    if config.SIM and gb.appMode != 'Calibration':
        analogOutput.connect(analogInput, [0,1,2,3])
        if gb.session.recording.value == 'Physiology':
            physiologyOutput = daq.AnalogOutput('/%s/ao1:%d' % (config.DAQ_B,
                config.ELECTRODE_COUNT), 31.25e3, np.inf, dataChunk=2,
                name='physiologyOutput', dataNeeded=physiologyOutputDataNeeded)
            physiologyOutput.connect(physiologyInput,
                list(range(config.ELECTRODE_COUNT)))

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
    data = np.random.randn(task.channelCount, nsNeeded)
    t = np.arange(nsWritten, nsWritten+nsNeeded) / task.fs
    data += np.sin(2*np.pi*1*t)*10
    data += np.sin(2*np.pi*100*t)*.5
    data += np.sin(2*np.pi*500*t)*.5
    return data
