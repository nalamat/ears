'''Internal software constants and configurations.


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
import logging


SIM               = '--sim' in sys.argv

APP_NAME          = 'EARS'
APP_FULL_NAME     = 'Electrophysiology Auditory Recording System'

# formats
DATETIME_FMT_LESS = '%Y%m%d-%H%M%S'
DATETIME_FMT_MORE = '%Y/%m/%d-%H:%M:%S'
LOG_FORMAT        = '[%(asctime)s.%(msecs)03d, %(module)s.py, ' + \
                    '%(funcName)s, %(levelname)s] %(message)s'
LOG_LEVEL         = logging.INFO

# file extensions
LOG_EXT           = '.log'
DATA_EXT          = '.h5'
SETTINGS_EXT      = '.json'
STIM_EXT          = '.wav'
DATA_FILTER       = 'HDF5 (*%s)' % DATA_EXT
SETTINGS_FILTER   = 'JSON (*%s)' % SETTINGS_EXT

# directories and files
ICONS_DIR         = 'icons/'
STIM_DIR          = 'stim/'
LOG_DIR           = os.path.expanduser(
                    '~/.%s/' % APP_NAME.lower()).replace('\\', '/')
LAST_SESSION_FILE = LOG_DIR   + 'last-session' + SETTINGS_EXT
APP_LOGO          = ICONS_DIR + 'logo.svg'

# interface colors and styles
COLOR_GO          = (100, 200, 110)    # light green
COLOR_GOREMIND    = (220, 220,   0)    # yellow
COLOR_NOGO        = (255, 100, 100)    # light red
COLOR_NOGOREMIND  = (255, 140,   0)    # orange
COLOR_NOGOREPEAT  = (220,  30,  30)    # red
COLOR_MAP         = {'Go':COLOR_GO, 'Go remind':COLOR_GOREMIND,
                    'Nogo':COLOR_NOGO, 'Nogo remind':COLOR_NOGOREMIND,
                    'Nogo repeat':COLOR_NOGOREPEAT}

COLOR_SPEAKER     = (255,   0,   0)
COLOR_MIC         = (  0,   0,   0)
COLOR_POKE        = (  0, 200,   0)
COLOR_SPOUT       = (  0,   0, 200)
COLOR_BUTTON      = (100, 100, 100)
COLOR_TRIAL       = (139,  69,  19, 100)
COLOR_TARGET      = (  0, 255,   0, 100)
COLOR_PUMP        = (  0,   0, 255, 100)
COLOR_TIMEOUT     = (  0,   0,   0, 100)

BORDER_STYLE      = '1px solid rgb(185,185,185)'

DAQ_A             = 'dev1'   # all analog/digital input/output except physiology
DAQ_B             = 'dev2'   # only physiology analog input
ELECTRODE_COUNT   = 15
