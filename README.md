# EARS: Electrophysiology Auditory Recording System

The EARS software controls a closed-loop system developed for synchronized
rodent auditory behavioral assessment with simultaneous real-time wireless
recording of cortical electrophysiology to study hearing in situations with
background sound. EARS supports data acquisition with up to two simultaneous
sound sources, a target and a masker. The masker sound is played continuously
throughout each recording session. The target sound is triggered through animal
behavior and played back at a fixed phase delay relative to the masker. The
signal-to-noise ratio of target and masker can be adjusted online. General
digital/analog input/output operations are facilitated through two PCIe data
acquisition (DAQ) cards (PCIe-6321 and PCIe-6341, National Instruments
Corporation). A single 16-bit analog channel is utilized at 100 kS/s to output
auditory stimuli to a sound amplifier and then into a loudspeaker. 15 analog
input channels are employed for collecting physiological signals from a
bioamplifier at a sampling frequency of 31.25 kS/s per channel. Custom
electronic circuits are designed to drive IR emitter diodes and also to
condition the output signal of their paired photodiodes (See Resources section
for circuit designs) so as to make compatible with the DAQ card’s digital input
channels. An additional digital channel is used to control the testing booth’s
lights. All input/output channels are synchronized by sharing the sampling clock
pulse of the two DAQ cards via an RTSI bus cable. A syringe pump (NE-1000
Programmable Single Syringe Pump, New Era Pump Systems, Inc.) is interfaced
through a USB-RS232 emulator using its own factory defined protocols and set of
commands.

This work was completed as a part of my PhD in the Neural Engineering of Speech
and Hearing (NESH) lab at New Jersey Institute of Technology (NJIT) and Rutgers
University, and was supported by funding from NIH NIDCD R03-DC014008.

Here's a video of the software in action:
[recording-1.mp4](media/recording-1.mp4?raw=true)


## How to use

Make sure all the packages listed in DEPENDENCIES.txt are installed.
The main entry point of the software is load.py:

    $ python load.py

Optionally, if no particular hardware (DAQ, pump, etc.) is connected,
a simulation mode is available for development and testing purposes:

    $ python load.py --sim

First the setup window opens for general experiment settings. Depending on the
selected recording type, either the behavior or both behavior and physiology
windows shown below will open for viewing and controlling the experiment
session.

![Behavior Window](media/screenshot-1.png?raw=true)

![Physiology Window](media/screenshot-2.png?raw=true)


## Resources

Inspired by and partially rewritten from NeuroBehavior by Brad Buran:
https://bitbucket.org/bburan/neurobehavior

Electronic circuits:
- Infrared sensor control: https://github.com/bburan/circuit-IR-sensor
- Low dropout power supply: https://github.com/nalamat/supply-ldo-adj-single
- Switching power supply: https://github.com/nalamat/supply-isr-adj-single


## License

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.


## Web and contact

Visit the EARS page at GitHub:
https://github.com/nalamat/ears

Ask questions, report bugs and give suggestions here:
https://github.com/nalamat/ears/issues

Visit lab website:
https://centers.njit.edu/nesh

Feel free to email us about anything at:
- NESH Lab: ears.software@gmail.com

Written by:
- Nima Alamatsaz: nima.alamatsaz@gmail.com
