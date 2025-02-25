
**Key points**

- The wavelength of a sound equals the speed divived by frequency

  - 20 Hz in water / 20 Hz in air

    - $$
      \frac{1500}{20} = 75 m \\

      \frac{360}{20} = 17 m
      $$
- State-of-the-art modulation schemes -> look **Table 3** to get inspiration of how we will modulate ours -> none of them use simple just amplitude modulation, so maybe its too simple

**Motivation for IoUT**

- 73% of the world i covered in water, the oceans are responsible for regulating temperature, the floara and fauna underwater also affects terristial life
- Underwater senors to measure changes in geological acitivity -> heads up for earthquakes or potential threats to aquatic life
- Millitary activities -> by improving data aqusitions underwater (via video)
- Underwater pollution
- Pipeline repair, ship wreck survey, fish farming activities

**IoUT communication processes**

- Either acoustic or RF (radio freq)
- Single-hop or multi-hop: hop referes to how many "hops" the data packet must go through to end at the final destination

  - Singlehop: sensor -> final destination
  - Multihop: sensor -> gateway device -> buoy -> final destination

**Components of IoUT**

* IoUT can be defined as be defined as underwater or buoy (surface) devices that are stationary or mobile (examples: submarines, ROV (remotely operated vechicle), sensors)
* Different IoUT devices are used for different things: sensor with imaging or video is short-term monitoring, tracking or exploration application (its prolly short term bcs video / image requires a lot of storage space etc

Below is a figures of an underwater sensing network in the most simple setup

<imgsrc="image/notes_for_above/1740475937141.png"alt="Description of the image "height = 350>

Slighty more complicated below

<imgsrc="image/notes_for_above/1740476028145.png"alt="Description of the image "height = 350>

**Challenges for IoUT**

- The main challenge lies in QoS (quality of service), which relates to throughput (how much data a system can process within a given time), delay, delay variation, data loss

  - Delay variation: acoustic waves exhibit random signal variation because of the motion of the water. This is because of the dopple effect, which causes freqency shifthing and spreading
  - Data rate: the amount of data being able to be transfered is dependent on the distance with which to travel (longer distances = lower bit rate)
  - Delay: at shallow waters ms of delay, deep waters up to seconds up deley -> this can cause ISI (inter-symbol interference)
  - Loss and bit-error: background noise in the water, absorptive loss -> when acoustic waves are made into heat energy, some of the energy is lost, geometric spreading -> the longer the distance, the more energy is lost, scattering loss -> waves with irregular surfaces and non-uniformaties makes the sound waves lose data
  - Multipath effect: the phenomenon when the same transmitted signal hits the receiver from different paths (reflection)
- The lack of knowledge of the water attributes (temp and depth) is the limiting factor for the QoS for acoutics waves, since the data transfer throught the medium (the water) is dependent on those factors

<imgsrc="image/notes_for_above/1740475680304.png"alt="Description of the image "height = 350>

Different ranges for acoustic data transfer, with their bandwidth etc

<imgsrc="image/notes_for_above/1740478192137.png"alt="Description of the image "height = 250>
