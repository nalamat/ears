# [EARS: Electrophysiology Auditory Recording System](github.com/nalamat/ears)

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

[![Live recording](media/recording-1.gif?raw=true)](media/recording-1.mp4?raw=true)


## How to use

Make sure all the packages listed in [REQUIREMENTS.txt](REQUIREMENTS.txt) are installed.
The main entry point of the software is [load.py](load.py):

    $ python load.py

Optionally, if no particular hardware (DAQ, pump, etc.) is connected,
a simulation mode is available for development and testing purposes:

    $ python load.py --sim

First the setup window opens for general experiment settings. Depending on the
selected recording type, either the behavior or both behavior and physiology
windows shown below will open for viewing and controlling the experiment
session.

![Behavior window screenshot](media/screenshot-1.png?raw=true)

![Physiology window screenshot](media/screenshot-2.png?raw=true)


## [Pypeline: Online stream processing](github.com/nalamat/pypeline)

The `Pypeline` module provides a generic, easy to use, and extendable object-oriented framework for online processing of data streams in Python, which is particularly suited for handling multi-channel electrophysiology signals in the EARS software.

Inspired by the `dplyr` package in R, `Pypeline` allows definition of stream processing stages as `Node`s that can be connected to each other using `>>`, the shift operator. Alternatively, `|` or the shell pipe operator can be used.

```python
daqs.physiologyInput \
    >> pypeline.LFilter(fl=300, fh=6e3, n=6) \
    >> self.physiologyPlot
```

Here, a key difference with `dplyr` is that `>>` only declares the connections in the pipeline, but no actual processing of data occurs at this statement. All `Node`s inherit the `write()` method that when called, passes new data into the node for processing. The processed data will further be passed to the connected nodes downstream.

Normally data is processed synchronously in the pipeline, meaning the execution of the code goes on halt until all downstream nodes are done with their tasks. Although, a `Thread` node can be inserted into the pipeline to allow asynchronous processing of the data. This could prove useful in situations that the data acquisition thread must not be blocked for too long or when implementing an interactive GUI. Note that in the current implementation of `Thread`, due to limitations of Python's GIL, multithreaded code does not actually run in parallel, but asynchronously. If necessary, the `wait()` method can be called on the root node to block execution until an asynchronous pipeline is done processing.

```python
daqs.physiologyInput \
    >> pypeline.Thread() \
    >> pypeline.LFilter(fl=300, fh=6e3, n=6) \
    >> pypeline.GrandAverage() \
    >> pypeline.DownsampleMinMax(ds=32) \
    >> self.physiologyPlot
```

Other than linear pipelines, it is possible to connect nodes to multiple branches. This is done by applying `>>` between a node and an iterator of nodes. If the left hand side of `>>` has a single node, its output data will be passed to each of the nodes in the iterator in the order of appearance. In the following example, the same signal is filtered at different frequency bands and then passed on to different plots.

```python
daqs.physiologyInput \
    >> pypeline.Thread() \
    >> (pypeline.LFilter(fl=None, fh=300 , n=6) >> self.physiologyPlotLow,
        pypeline.LFilter(fl=300 , fh=6e3 , n=6) >> self.physiologyPlotMid,
        pypeline.LFilter(fl=6e3 , fh=None, n=6) >> self.physiologyPlotHigh)
```

Visualizations of pipeline structures are coming soon!

If instead of passing the same data to all downstream nodes, a splitting behavior is required, use the `Split` node between the source and the list of nodes. The code below passes spikes detected from each channel of the recorded signal to separate plots:

```python
daqs.physiologyInput \
    >> pypeline.Thread() \
    >> pypeline.LFilter(fl=300 , fh=6e3 , n=6) \
    >> pypeline.SpikeDetector() \
    >> pipeline.Split() \
    >> (self.spikePlot1, self.spikePlot2, self.spikePlot3)
```


## [glPlotLib: GPU accelerated plotting](github.com/nalamat/glplotlib)

Currently in development.

For a demo of GPU accelerated plotting using OpenGL and online spike detection, try:

    $ python glPlotLib.py

![GPU accelerated plotting screenshot](media/screenshot-3.png?raw=true)


## Resources

Inspired by and partially rewritten from NeuroBehavior by Brad Buran:
<https://bitbucket.org/bburan/neurobehavior>

Electronic circuits:
- Infrared sensor control: <https://github.com/bburan/circuit-IR-sensor>
- Low dropout power supply: <https://github.com/nalamat/supply-ldo-adj-single>
- Switching power supply: <https://github.com/nalamat/supply-isr-adj-single>


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
<https://github.com/nalamat/ears>

Ask questions, report bugs and give suggestions here:
<https://github.com/nalamat/ears/issues>

Visit NESH Lab website:
<https://ihlefeldlab.com>

Written by:
- Nima Alamatsaz: <nima.alamatsaz@gmail.com>
