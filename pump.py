'''Communication with New Era pump series using a RS-232 interface.


This file is part of the EARS project: https://github.com/nalamat/ears
Copyright (C) 2017-2018 Nima Alamatsaz <nima.alamatsaz@njit.edu>
Copyright (C) 2017-2018 Antje Ihlefeld <antje.ihlefeld@njit.edu>
Distrubeted under GNU GPLv3. See LICENSE.txt for more info.

Original project repository: https://bitbucket.org/bburan/new-era
Copyright (C) 2011-2012 Brad Buran <bburan@alum.mit.edu>
'''

import re
import sys
import socket
import struct
import logging
import threading
from   select    import select


log = logging.getLogger(__name__)
SIM = '--sim' in sys.argv

if not SIM:
    import serial


def convert(value, src_unit, dest_unit):
    MAP = {
            ('ul',     'ml'):     lambda x: x*1e-3,
            ('ml',     'ul'):     lambda x: x*1e3,
            ('ul/min', 'ml/min'): lambda x: x*1e-3,
            ('ul/min', 'ul/h'):   lambda x: x*60.0,
            ('ul/min', 'ml/h'):   lambda x: x*60e-3,
            ('ml/min', 'ul/min'): lambda x: x*1e3,
            ('ml/min', 'ul/h'):   lambda x: x*60e3,
            ('ml/min', 'ml/h'):   lambda x: x*60,
            ('ul/h',   'ml/h'):   lambda x: x*1e-3,
            }
    if src_unit == dest_unit:
        return value
    return MAP[src_unit, dest_unit](value)

#####################################################################
# Custom-defined pump error messages
#####################################################################

class PumpError(Exception):
    '''
    General pump error
    '''

    def __init__(self, code, mesg=None):
        self.code = code
        self.mesg = mesg

    def __str__(self):
        result = '%s\n\n%s' % (self._todo, self._mesg[self.code])
        if self.mesg is not None:
            result += ' ' + self.mesg
        return result

class PumpCommError(PumpError):
    '''
    Handles error messages resulting from problems with communication via the
    pump's serial port.
    '''

    _mesg = {
            # Actual codes returned by the pump
            ''      : 'Command is not recognized',
            'NA'    : 'Command is not currently applicable',
            'OOR'   : 'Command data is out of range',
            'COM'   : 'Invalid communications packet received',
            'IGN'   : 'Command ignored due to new phase start',
            # Custom codes
            'NR'    : 'No response from pump',
            'SER'   : 'Unable to open serial port',
            'UNK'   : 'Unknown error',
            }

    _todo = 'Unable to connect to pump.  Please ensure that no other ' + \
            'programs that utilize the pump are running and try ' + \
            'try power-cycling the entire system (rack and computer).'

class PumpHardwareError(PumpError):

    '''Handles errors specific to the pump hardware and firmware.'''

    _mesg = {
            'R'     : 'Pump was reset due to power interrupt',
            'S'     : 'Pump motor is stalled',
            'T'     : 'Safe mode communication time out',
            'E'     : 'Pumping program error',
            'O'     : 'Pumping program phase out of range',
            }

    _todo = 'Pump has reported an error.  Please check to ensure pump ' + \
            'motor is not over-extended and power-cycle the pump.'

class PumpUnitError(Exception):
    '''Occurs when the pump returns a value in an unexpected unit
    '''

    def __init__(self, expected, actual, cmd):
        self.expected = expected
        self.actual = actual
        self.cmd = cmd

    def __str__(self):
        mesg = '%s: Expected units in %s, receved %s'
        return mesg % (self.cmd, self.expected, self.actual)

class PumpInterface(object):
    '''
    Establish a connection with the New Era pump.
    '''

    #####################################################################
    # Basic information required for creating and parsing RS-232 commands
    #####################################################################

    # Hex command characters used to indicate state of data
    # transmission between pump and computer.
    ETX = '\x03'    # End of packet transmission
    STX = '\x02'    # Start of packet transmission
    CR  = '\x0D'    # Carriage return

    # The Syringe Pump uses a standard 8N1 frame with a default baud rate of
    # 19200.  These are actually the default parameters when calling the command
    # to init the serial port, but I define them here for clarity (especially if
    # they ever change in the future).
    CONNECTION_SETTINGS = dict(baudrate=19200, bytesize=8, parity='N',
            stopbits=1, timeout=.05, xonxoff=0, rtscts=0, writeTimeout=1,
            dsrdtr=None, interCharTimeout=None)

    STATUS = dict(I='infusing', W='withdrawing', S='halted', P='paused',
                  T='in timed pause', U='waiting for trigger', X='purging')

    # Map of trigger modes.  Dictionary key is the value that must be provided
    # with the TRG command sent to the pump.  Value is a two-tuple indicating
    # the start and stop trigger for the pump (based on the TTL input).  The
    # trigger may be a rising/falling edge, a low/high value or None.  If you
    # set the trigger to 'falling', None', then a falling TTL will start the
    # pump's program with no stop condition.  A value of 'rising', 'falling'
    # will start the pump when the input goes high and stop it when the input
    # goes low.
    TRIG_MODE = {
            'FT':   ('falling', 'falling'),
            'FH':   ('falling', 'rising'),
            'F2':   ('rising',  'rising'),
            'LE':   ('rising',  'falling'),
            'ST':   ('falling', None),
            'T2':   ('rising',  None),
            'SP':   (None,      'falling'),
            'P2':   (None,      'falling'),
            'RL':   ('low',     None),
            'RH':   ('high',    None),
            'SL':   (None,      'low'),
            'SH':   (None,      'high'),
            }

    REV_TRIG_MODE = dict((v, k) for k, v in TRIG_MODE.items())

    DIR_MODE = {
            'INF':  'infuse',
            'WDR':  'withdraw',
            'REV':  'reverse',
            }

    REV_DIR_MODE = dict((v, k) for k, v in DIR_MODE.items())

    RATE_UNIT = {
            'UM':   'ul/min',
            'MM':   'ml/min',
            'UH':   'ul/h',
            'MH':   'ml/h',
            }

    REV_RATE_UNIT = dict((v, k) for k, v in RATE_UNIT.items())

    VOL_UNIT = {
            'UL':   'ul',
            'ML':   'ml',
            }

    REV_VOL_UNIT = dict((v, k) for k, v in VOL_UNIT.items())

    # The response from the pump always includes a status flag which indicates
    # the pump state (or error).  Response is in the format
    # <STX><address><status>[<data>]<ETX>
    _basic_response = re.compile(STX + '(?P<address>\d+)' + \
                                       '(?P<status>[IWSPTUX]|A\?)' + \
                                       '(?P<data>.*)' + ETX)

    # Response for queries about volume dispensed.  Returns separate numbers for
    # infuse and withdraw.  Format is I<float>W<float><units>
    _dispensed = re.compile('I(?P<infuse>[\.0-9]+)' + \
                            'W(?P<withdraw>[\.0-9]+)' + \
                            '(?P<units>[MLU]{2})')

    #####################################################################
    # Special functions for controlling pump
    #####################################################################

    def __init__(self, start_trigger='rising', stop_trigger='falling',
            volume_unit='ml', rate_unit='ml/min', port='COM1'):

        if SIM: return

        self._port = port
        self._lock = threading.Lock()
        self.connect()

        # We do not currently support changing the units of the pump on-the-fly.
        # They must be initialized here.
        self.rate_unit       = rate_unit
        self.volume_unit     = volume_unit
        self.rate_unit_cmd   = self.REV_RATE_UNIT[rate_unit]
        self.volume_unit_cmd = self.REV_VOL_UNIT [volume_unit]
        self._lastInfused    = 0
        self._lastWithdrawn  = 0
        self._xmit('VOL %s' % self.volume_unit_cmd)
        self.setTrigger(start=start_trigger, stop=stop_trigger)
        log.debug('Connected to pump %s', self._xmit('VER'))

    def connect(self):
        if SIM: return

        try:
            # Connection is shared across all classes.  Raise warning if we
            # create a new instance?
            if not hasattr(self, 'ser'):
                cn = serial.Serial(port=self._port, **self.CONNECTION_SETTINGS)
                PumpInterface.ser = cn
            if not self.ser.isOpen():
                self.ser.open()

            # Pump baudrate must match connection baudrate otherwise we won't be
            # able to communicate
            self._xmit('ADR 0 B %d' % self.CONNECTION_SETTINGS['baudrate'])
            # Turn audible alarm on.  This will notify the user of any problems
            # with the pump.
            self._xmit('AL 0')
            # Ensure that serial port is closed on system exit
            import atexit
            atexit.register(self.disconnect)
        except PumpHardwareError as e:
            # We want to trap and dispose of one very specific exception code,
            # 'R', which corresponds to a power interrupt.  This is almost
            # always returned when the pump is first powered on and initialized
            # so it really is not a concern to us.  The other error messages are
            # of concern so we reraise them.
            if e.code != 'R':
                raise
        except NameError as e:
            # Raised when it cannot find the global name 'SERIAL' (which
            # typically indicates a problem connecting to COM1).  Let's
            # translate this to a human-understandable error.
            log.exception(e)
            raise PumpCommError('SER')

    def disconnect(self):
        '''Stop pump and close serial port.

        Automatically called when Python exits.
        '''
        if SIM: return

        try:
            self.stop()
        finally:
            self.ser.close()
            return # Don't reraise error conditions, just quit silently

    def start(self):
        '''
        Starts the pump.
        '''
        if SIM: return

        with self._lock:
            self._xmit('RUN')

    def startIfTTL(self, value=True):
        '''
        In contrast to `run`, the logical state of the TTL input is inspected
        (high=True, low=False).  If the TTL state is equal to value, the pump
        program is started.

        If value is True, start only if the TTL is high.  If value is False,
        start only if the TTL is low.
        '''
        if SIM: return

        if self.getTTL() == value:
            self.start()

    def stop(self):
        '''Stop the pump. Raises PumpError if the pump is already stopped.'''
        if SIM: return

        with self._lock:
            self._xmit('STP')

    def pause(self):
        if SIM: return

        self._trigger = self.getTrigger()
        self.setTrigger(None, 'falling')
        try:
            self.stop()
        except PumpError:
            pass

    def resume(self):
        if SIM: return

        self.setTrigger(*self._trigger)
        if self._trigger[0] in ('high', 'rising'):
            self.runIfTTL(True)
        elif self._trigger[0] in ('low', 'falling'):
            self.runIfTTL(False)

    def setTrigger(self, start, stop):
        '''
        Set the start and stop trigger modes.  Valid modes are rising, falling,
        high and low.  Note that not all combinations of modes are supported
        (see TRIG_MODE for supported pairs).

        start=None, stop='falling': pump program stops on a falling edge (start
        manually or use the `run` method to start the pump)

        start='rising', stop='falling': pump program starts on a rising edge and
        stops on a falling edge
        '''
        if SIM: return

        with self._lock:
            cmd = self.REV_TRIG_MODE[start, stop]
            self._xmit('TRG %s' % cmd)

    def getTrigger(self):
        '''
        Get trigger mode.  Returns tuple of two values indicating start and stop
        condition.
        '''
        if SIM: return

        with self._lock:
            value = self._xmit('TRG')
            return self.TRIG_MODE[value]

    def setDirection(self, direction):
        '''Set direction of the pump.

        Valid directions are 'infuse', 'withdraw' and 'reverse'.
        '''
        if SIM: return

        with self._lock:
            arg = self.REV_DIR_MODE[direction]
            self._xmit('DIR %s' % arg)

    def getDirection(self):
        '''Get current direction of the pump.

        Response will be either 'infuse' or 'withdraw'.
        '''
        if SIM: return

        with self._lock:
            value = self._xmit('DIR')
            return self.DIR_MODE[value]

    def getRate(self, unit=None):
        '''
        Get current rate of the pump, converting rate to requested unit.  If no
        unit is specified, value is in the units specified when the interface
        was created.
        '''
        if SIM: return

        with self._lock:
            value = self._xmit('RAT')
            if value[-2:] != self.rate_unit_cmd:
                raise PumpUnitError(self.volume_unit_cmd, value[-2:])
            value = float(value[:-2])
            if unit is not None:
                value = convert(value, self.rate_unit, unit)
            return value

    def setRate(self, rate, unit=None):
        '''
        Set current rate of the pump, converting rate from specified unit to the
        unit specified when the interface was created.
        '''
        if SIM: return

        with self._lock:
            if unit is not None:
                rate = convert(rate, unit, self.rate_unit)
            self._xmit('RAT %0.3f %s' % (rate, self.rate_unit_cmd))

    def setVolume(self, volume, unit=None):
        '''
        Set current volume of the pump, converting volume from specified unit to
        the unit specified when the interface was created.
        '''
        if SIM: return

        with self._lock:
            if unit is not None:
                volume = convert(volume, unit, self.volume_unit)
            self._xmit('VOL %0.3f' % volume)

    def getVolume(self, unit=None):
        '''
        Get current volume of the pump, converting volume to requested unit.  If
        no unit is specified, value is in the units specified when the interface
        was created.
        '''
        if SIM: return

        with self._lock:
            value = self._xmit('VOL')
            if value[-2:] != self.volume_unit_cmd:
                raise PumpUnitError(self.volume_unit_cmd, value[-2:])
            value = float(value[:-2])
            if unit is not None:
                value = convert(value, unit, self.volume_unit)
            return value

    def _getDispensed(self, direction, unit=None):
        if SIM: return

        # Helper method for _getInfused and _getWithdrawn
        result = self._xmit('DIS')
        log.debug('_dispensed: %s, result: %s', self._dispensed, result)
        match = self._dispensed.match(result)
        if match.group('units') != self.volume_unit_cmd:
            raise PumpUnitError('ML', match.group('units'), 'DIS')
        else:
            value = float(match.group(direction))
            if unit is not None:
                value = convert(value, self.volume_unit, unit)
            return value

    def resetDispensed(self):
        '''Reset the cumulative infused/withdrawn volume.'''
        if SIM: return


        with self._lock:
            self._lastInfused   = 0
            self._lastWithdrawn = 0
            self._xmit('CLD INF')
            self._xmit('CLD WDR')

    def getInfused(self, unit=None):
        '''
        Get current volume withdrawn, converting volume to requested unit.  If
        no unit is specified, value is in the units specified when the interface
        was created.
        '''
        if SIM: return 0

        with self._lock:
            return self._getDispensed('infuse', unit) + self._lastInfused

    def getWithdrawn(self, unit=None):
        '''
        Get current volume dispensed, converting volume to requested unit.  If
        no unit is specified, value is in the units specified when the interface
        was created.
        '''
        if SIM: return

        with self._lock:
            return self._getDispensed('withdraw', unit) + self._lastWithdrawn

    def setDiameter(self, diameter, unit=None):
        '''Set diameter (unit must be mm).'''
        if SIM: return

        with self._lock:
            self._lastInfused   += self._getDispensed('infuse'  )
            self._lastWithdrawn += self._getDispensed('withdraw')
            if unit is not None and unit != 'mm':
                raise PumpUnitError('mm', unit, 'DIA')
            self._xmit('DIA %.2f' % diameter)

    def getDiameter(self):
        '''Get diameter setting in mm.'''
        if SIM: return

        with self._lock:
            return self._xmit('DIA')

    def getTTL(self):
        '''Get status of TTL trigger.'''
        if SIM: return

        with self._lock:
            data = self._xmit('IN 2')
            if data == '1':
                return True
            elif data == '0':
                return False
            else:
                raise PumpCommError('', 'IN 2')

    def getStatus(self):
        with self._lock:
            return self.STATUS[self._getRawResponse('')['status']]

    #####################################################################
    # RS232 functions
    #####################################################################

    def _readline(self):
        # PySerial v2.5 no longer supports the eol parameter, so we manually
        # read byte by byte until we reach the line-end character.  Timeout
        # should be set to a very low value as well.  A support ticket has been
        # filed (and labelled WONTFIX).
        # https://sourceforge.net/tracker/?
        # func=detail&atid=446302&aid=3101783&group_id=46487
        result = []
        while 1:
            last = self.ser.read(1).decode()
            result.append(last)
            if last == self.ETX or last == '':
                break
        return ''.join(result)

    def _xmitSequence(self, *commands):
        '''
        Transmit sequence of commands to pump and return list of responses to
        each command
        '''
        return [self._xmit(cmd) for cmd in commands]

    def _getRawResponse(self, command):
        self._send(command)
        result = self._readline()
        if result == '':
            raise PumpCommError('NR', command)
        match = self._basic_response.match(result)
        if match is None:
            raise PumpCommError('NR')
        if match.group('status') == 'A?':
            raise PumpHardwareError(match.group('data'), command)
        elif match.group('data').startswith('?'):
            raise PumpCommError(match.group('data')[1:], command)
        return match.groupdict()

    def _xmit(self, command):
        '''
        Transmit command to pump and return response

        All necessary characters (e.g. the end transmission flag) are added to
        the command when transmitted, so you only need to provide the command
        string itself (e.g. "RAT 3.0 MM").

        The response packet is inspected to see if the pump has an error
        condition (e.g. a stall or power reset).  If so, the appropriate
        exception is raised.
        '''
        return self._getRawResponse(command)['data']

    def _send(self, command):
        self.ser.write((command + self.CR).encode('ascii'))
