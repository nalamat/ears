'''Setup and load an experiment with user selectable settings.


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
import sys
import ctypes
import logging
import platform
import warnings
import traceback
from   PyQt5       import QtWidgets, QtGui

warnings.filterwarnings('ignore', category=FutureWarning, module='h5py')

import hdf5
import setup
import config
import behavior
import guiHelper
import physiology
import calibration
import globals     as gb


try:
    # change working directory to project root
    # keep old working directory and restore it later
    oldCWD = os.getcwd()
    newCWD = os.path.dirname(__file__)
    newCWD = os.path.realpath(newCWD)
    os.chdir(newCWD)

    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)

    # configure logging
    formatter = logging.Formatter(config.LOG_FORMAT,
        config.DATETIME_FMT_MORE)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    # setup first log file in user home directory
    fileHandler = logging.FileHandler(config.LOG_DIR +
        gb.sessionTimeCompact + config.LOG_EXT)
    fileHandler.setFormatter(formatter)
    rootLogger = logging.getLogger()
    rootLogger.addHandler(consoleHandler)
    rootLogger.addHandler(fileHandler)
    rootLogger.setLevel(config.LOG_LEVEL)

    # get logger for current module
    log = logging.getLogger(__name__)
    log.info('Opening application')

    # global exception handling
    def masterExceptionHook(exctype, value, traceback):
        if exctype in (SystemExit, KeyboardInterrupt):
            log.info('Exit requested with "%s"' % exctype.__name__)
        else:
            log.exception('Uncaught exception occured',
                exc_info=(exctype, value, traceback))
        hdf5.close()
        # restore old working directory
        os.chdir(oldCWD)
        log.info('Exiting application')
        sys.exit(1)
    sys._excepthook = sys.excepthook
    sys.excepthook  = masterExceptionHook

    # initialize app
    app = QtWidgets.QApplication([])
    app.setApplicationName = config.APP_NAME
    app.setWindowIcon(QtGui.QIcon(config.APP_LOGO))
    # without this app icon will not show in windows
    # see: https://stackoverflow.com/questions/1551605
    if platform.system() == 'Windows':
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            'nesh.' + config.APP_NAME.lower())

    # load last session settings, if any
    if os.path.isfile(config.LAST_SESSION_FILE):
        log.info('Loading last session settings from "%s"',
            config.LAST_SESSION_FILE)
        gb.session.loadFile(config.LAST_SESSION_FILE)

    # open session setup window
    log.info('Opening setup window')
    setupWindow = setup.SetupWindow()
    guiHelper.centerWindow(setupWindow)
    res = setupWindow.exec_()

    if res != QtWidgets.QDialog.Accepted:
        exit()

    if gb.session.calibrationFile.value:
        log.info('Loading calibration from "%s"',
        gb.session.calibrationFile.value)
        gb.calibration.loadFile(gb.session.calibrationFile.value, float)

    if gb.appMode == 'Calibration':
        log.info('Opening calibration window')
        calibrationWindow = calibration.CalibrationWindow()
        guiHelper.centerWindow(calibrationWindow)
        calibrationWindow.show()

    elif gb.appMode == 'Experiment':
        # setup second log file in the subject's own data directory
        dir_, file = os.path.split(gb.session.dataFile.value)
        name, ext  = os.path.splitext(file)
        logDir     = os.path.join(dir_, 'Logs')
        if not os.path.exists(logDir):
            os.makedirs(logDir)
        fileHandler = logging.FileHandler(
            os.path.join(logDir, name + config.LOG_EXT) )
        fileHandler.setFormatter(formatter)
        rootLogger.addHandler(fileHandler)

        if config.SIM:
            log.info('Simulation mode enabled')

        # load paradigm settings from file, if any
        # else, initialize selected rove paramater values
        if gb.session.paradigmFile.value:
            gb.paradigm.loadFile(gb.session.paradigmFile.value)
            # update session rove params with paradigm's
            gb.session.rove.value = gb.paradigm.rove.keys
        else:
            # init paradigm rove values for selected params
            gb.paradigm.rove.value = {}
            for name in gb.session.rove.value:
                item = gb.paradigm[name]
                gb.paradigm.rove.value[name] = [str(item.type())]*3

        # save current session settings
        gb.session.saveFile(config.LAST_SESSION_FILE)
        # prepare HDF5 data file
        hdf5.open(gb.session.dataFile.value, mode='w')

        # open behavior window
        log.info('Opening behavior window')
        behaviorWindow = behavior.BehaviorWindow()
        behaviorWindow.show()
        guiHelper.centerWindow(behaviorWindow)

        if gb.session.recording.value == 'Physiology':
            log.info('Opening physiology window')
            physiologyWindow = physiology.PhysiologyWindow()
            physiologyWindow.show()
            guiHelper.centerWindow(physiologyWindow, -2)

    app.exec_()

except (SystemExit, KeyboardInterrupt) as e:
    msg = 'Exit requested with "%s"' % type(e).__name__
    if 'log' in dir() and log:
        log.info(msg)
    else:
        print(msg)

except:
    msg = 'Coudn\'t initialize application'
    if 'log' in dir() and log:
        log.exception(msg)
    else:
        print(msg)
        traceback.print_exc()

    if 'app' in dir() and app:
        guiHelper.showException()

finally:
    hdf5.close()
    # restore old working directory
    os.chdir(oldCWD)

    if 'log' in dir() and log:
        log.info('Exiting application')
    else:
        print('Exiting application')
