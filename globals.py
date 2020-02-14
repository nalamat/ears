'''Definition of contexts accessible from all modules.


This file is part of the EARS project: https://github.com/nalamat/ears
Copyright (C) 2017-2020 Nima Alamatsaz <nima.alamatsaz@gmail.com>
Copyright (C) 2017-2020 NESH Lab <ears.software@gmail.com>
Distrubeted under GNU GPLv3. See LICENSE.txt for more info.
'''

import enum
import datetime as     dt

import config
from   context  import Item, Context


# can be either 'Experiment' or 'Calibration'
# value is set by setup.SetupWindow
appMode             = 'Experiment'

sessionTime         = dt.datetime.now()
sessionTimeCompact  = sessionTime.strftime(config.DATETIME_FMT_LESS)
sessionTimeComplete = sessionTime.strftime(config.DATETIME_FMT_MORE)

experimentStart     = None

behaviorWindow      = None
physiologyWindow    = None

syringeDiameters    = {'B-D 10cc': 14.43, 'B-D 20cc': 19.05,
    'B-D 30cc': 21.59, 'B-D 60cc': 26.59}

trialTypesGo        = ['Go remind', 'Go']
trialTypesNogo      = ['Nogo', 'Nogo remind', 'Nogo repeat']
trialTypes          = trialTypesGo + trialTypesNogo
trialStatesAwaiting = ['None', 'Awaiting random trigger', 'Awaiting spout',
    'Awaiting button', 'Awaiting poke']

'''General session settings.'''
session = Context(
    Item('dataDir'           , 'Data directory'   , str  , 'data'            ),
    Item('subjectID'         , 'Subject ID'       , str  , 'C1Fluffy'        ),
    Item('experimentName'    , 'Experiment name'  , str  , 'MMR'              ,
        ['MMR', 'EM', 'IM']                                                  ),
    Item('experimentMode'    , 'Experiment mode'  , str  , 'Go Nogo'          ,
        ['Passive', 'Spout training', 'Target training' , 'Poke training'    ,
        'Go Nogo']                                                           ),
    Item('recording'         , 'Recording'        , str  , 'Behavior'         ,
        ['Behavior', 'Physiology']                                           ),
    Item('autoDataFile'      , 'Auto data file'   , bool , True              ),
    Item('dataFile'          , 'Data file'        , str  , ''                ),
    Item('calibrationFile'   , 'Calibration file' , str  , ''                ),
    Item('paradigmFile'      , 'Paradigm file'    , str  , ''                ),
    Item('rove'              , 'Rove params'      , list ,
        ['targetFrequency', 'targetLevel']                                   ),
    Item('sessionTime'       , ''                 , str  , ''                ),
    Item('experimentStart'   , ''                 , str  , ''                ),
    Item('experimentStop'    , ''                 , str  , ''                ),
    Item('experimentDuration', ''                 , float, 0                 ),
    Item('totalReward'       , ''                 , float, 0                 ),
    Item('dataViable'        , 'Data viable'      , bool , False             ),
    Item('notes'             , 'Notes'            , str  , ''                ),
    Item('computerName'      , ''                 , str  , ''                ),
    Item('commitHash'        , ''                 , str  , ''                ),
    )

'''Calibration settings.

Values specify the calibrated dB SPL of a sound file at 0 dB amplification.
'''
calibration = Context(
    # Item('Supermasker.wav', '', float, 80),
    )

'''Paradigm settings.

Args:
    rove (dict of list of str): Each dict key is the name of a rove parameter
        and each value is a list of str. First element of the list is used for
        the nogo condition, second for remind and the rest for go trials.
        Example: {'level':['0','75','50','60'], 'freq':['0','1','1','2']}
               level   freq
        Nogo     0      0
        Remind  75      1
        Go      50      1
        Go      60      2
'''
paradigm = Context(
    Item('rove'               , ''                         , dict , {}        ),

    Item('group'              , 'Constant limits'                             ),
    Item('goProbability'      , 'Go probability'           , float, '0.75'    ),
    Item('goOrder'            , 'Go order'                 , str  , 'Random'   ,
        ['Random', 'Pseudorandom', 'Exact order', 'Reverse order']            ),
    Item('repeatFA'           , 'Repeat if FA'             , bool , True      ),

    Item('group'              , 'Stimulus'                                    ),
    # Item('targetType'         , 'Target type'              , str  , 'Tone'   ,
    #     ['Tone', 'File']                                                    ),
    # Item('targetFrequency'    , 'Target frequency (kHz)'   , float, '1.0'   ),
    Item('targetFile'         , 'Target file'              , str  , 'T01.wav' ),
    Item('targetLevel'        , 'Target level (dB SPL)'    , float, '75.0'    ),
    Item('targetDuration'     , 'Target duration (s)'      , float, '1.0'     ),
    Item('targetRamp'         , 'Target ramp (ms)'         , float, '50'      ),
    Item('maskerFile'         , 'Masker file'              , str  , ''        ),
    Item('maskerLevel'        , 'Masker level (dB SPL)'    , float, '50.0'    ),
    Item('maskerFrequency'    , 'Masker frequency (Hz)'    , float, '10.0'    ),

    Item('group'              , 'Timing'                                      ),
    Item('phaseDelay'         , 'Phase delay (deg)'        , float, '45'      ),
    Item('minPokeDuration'    , 'Min poke duration (s)'    , float, '0.2'     ),
    Item('pokeHoldDuration'   , 'Poke hold duration (s)'   , float, '0.0'     ),
    Item('holdDuration'       , 'Hold duration (s)'        , float, '0.2'     ),
    Item('maxResponseDuration', 'Max response duration (s)', float, '4.0'     ),
    Item('timeoutDuration'    , 'Timeout duration (s)'     , float, '1.5'     ),
    Item('intertrialDuration' , 'Intertrial duration (s)'  , float, '0.1'     ),

    Item('group'              , 'Syringe'                                     ),
    Item('pumpRate'           , 'Pump rate (ml/min)'       , float, '1.5'     ),
    Item('rewardVolume'       , 'Reward volume (ul)'       , float, '25'      ),
    Item('syringeType'        , 'Syringe type'             , str  , 'B-D 60cc' ,
        list(syringeDiameters.keys())                                         ),
    )

'''Current or upcoming trial context.

Args:
    roveValues (str): Comma seperated string of evaluated rove values. Used
        when showing trial log and calculating performance.
'''
trial = Context(
    # Item('roveValues'      , 'Rove\nvalue(s)'    , str  ),
    Item('trialType'       , ''                  , str   ,
        values=trialTypes                               ),
    Item('trialStart'      , 'Trial\nstart'      , float),
    Item('trialStop'       , ''                  , float),
    Item('pokeStart'       , ''                  , float),
    Item('pokeStop'        , ''                  , float),
    Item('targetStart'     , ''                  , float),
    Item('targetStop'      , ''                  , float),
    Item('spoutStart'      , ''                  , float),
    # Item('spoutStop'       , type=float                 ),
    Item('pumpStart'       , ''                  , float),
    Item('pumpStop'        , ''                  , float),
    Item('timeoutStart'    , ''                  , float),
    Item('timeoutStop'     , ''                  , float),
    Item('pokeDuration'    , 'Poke\nduration'    , float),
    Item('responseDuration', 'Response\nduration', float),
    Item('score'           , 'Score'             , str   ,
        values=['None', 'HIT', 'MISS', 'CR', 'FA']      ),    # TODO: fix cases?
    Item('response'        , ''                  , str   ,
        values=['None', 'Poke', 'Spout']                ),
    )

paradigm.copyTo(trial, copyTypeless=False, copyLabel=False)
trial.clearValues()

'''Current status of the experiment.'''
status = Context(
    Item('ts'             , 'Time (min:sec)'   , str, '00:00.00'   , link=True),
    Item('fps'            , 'Frame rate (fps)' , str, '0.0'        , link=True),
    Item('experimentState', 'Experiment state' , str, 'Not started', link=True ,
        values=['Not started', 'Running', 'Paused']                           ),
    Item('trialState'     , 'Trial state'      , str, 'None'       , link=True ,
        values=['None',
        'Awaiting random trigger', 'Trial ongoing',            # passive
        'Awaiting spout', 'Spout active',                      # spout training
        'Awaiting button','Target active','Response duration', # target trianing
        'Awaiting poke', 'Min poke duration',          # poke trianing & go nogo
        'Poke hold duration', 'Hold duration', 'Response duration',
        'Timeout duration', 'Intertrial duration']                            ),
    Item('trialType'      , 'Trial type'       , str, 'Go remind'  , link=True ,
        values=trialTypes                                                     ),
    Item('totalReward'    , 'Total reward (ml)', str, '0.000'      , link=True),
    )

# performance parameters with empty label will not be shown in performance table
# note: count types are specified as float to allow setting to np.nan
performance = Context(
    # Item('roveValues'      , 'Rove\nvalue(s)'    , str  ),
    Item('trialType'       , ''                  , str   ,
        values=trialTypes                               ),
    Item('trialCount'      , 'Trial\ncount'      , float),
    Item('hitCount'        , ''                  , float),
    Item('missCount'       , ''                  , float),
    Item('crCount'         , ''                  , float),
    Item('faCount'         , ''                  , float),
    Item('hitRate'         , 'HIT\nrate'         , float),
    Item('faRate'          , 'FA\nrate'          , float),
    Item('pokeDuration'    , 'Poke\nduration'    , float),
    Item('responseDuration', 'Response\nduration', float),
    Item('dPrime'          , 'd\''               , float),
    )

if __name__ == '__main__':
    trial.initData()
    trial.trialType.value = 'Go remind'
    trial.appendData()
    trial.trialType.value = 'Go'
    trial.appendData()
    trial.trialType.value = 'Nogo'
    trial.appendData()
    trial.trialType.value = 'Nogo repeat'
    trial.appendData()
    trial.trialType.value = 'Nogo'
    trial.appendData()
