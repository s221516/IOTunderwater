Voltage: Voltage is the force that makes electrons flow. It's a difference in potential energy between two different points in a circuit.
Current: Current is the rate of the flow of electrons. It's measured in amperes, which are also called amps.

How does the waves from the signal generator and output as AC vaguely relate?

The signal generator outputs AC (alternating current), which means the voltage alternates back and forth, changing its polarity over time (positive to negative and back again). If you were to measure the output with a voltmeter, the voltage would constantly flip between positive and negative values.

This alternating behavior follows a specific waveform, which describes how the voltage changes over time. For example:

A sine wave changes smoothly, rising and falling in a continuous curve.
A square wave alternates abruptly, jumping between high and low voltages.
In essence, the signal generator creates these voltage waves, which can be used to drive devices like speakers or other circuits.

Amplitude of voltage wave:

- 10mVpp to 10 Vpp (means the smallest peaks are 10mVpp and the largest are 10 Vpp)

Freq of the voltages wave:

- how fast/often the electrons swap direction, essentially resulting in normal wave frequency
- <10MHz very accurate

Offset

- add constant DC to the signal so it is always shifted

Impedance
---------

Why Does the Speaker Draw Less Current in This Mismatch?
The confusion here is about how current flows in a circuit with impedance mismatch.

Lower Impedance of the Load (Speaker) means the total impedance of the system (generator + speaker) is lower. The total impedance is determined by the combination of the signal generator's 50Ω impedance and the speaker's 8Ω impedance.

Ohm's Law (V = IR) tells us that the current (I) is determined by the voltage (V) provided by the generator and the total impedance (R) of the circuit.

𝐼
==

𝑉
𝑅
total
I=
R
total

V

Since total impedance is lower (because the speaker is only 8Ω), the current will be higher in the circuit.

But here’s the key: The signal generator wasn't designed to drive a lower impedance load (like 8Ω), so it can't provide as much current efficiently. In fact, it might output less current overall, and the speaker won't get enough power.

So, What Actually Happens?
Less Voltage to the Speaker: Because of the impedance mismatch, the generator won't be able to transfer all of its voltage to the speaker. The 8Ω speaker will receive less voltage than it would if it were connected to an ideal 8Ω source.

Result: The current to the speaker is less than it could be if the system were matched. So while Ohm's Law says current increases with a lower load impedance, the generator itself limits current because it can't handle the extra current demand efficiently. Therefore, the speaker doesn't receive as much current as it could in a matched scenario.

In Short:

The speaker draws more current than the generator is designed to provide, but the generator can't supply enough current to make the speaker perform at its full potential.
The speaker doesn’t receive enough power because of the voltage drop caused by the impedance mismatch.

cock
