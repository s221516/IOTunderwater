"A basic underwater communication system that can transmit data from an underwater speaker to a hydrophone receiver".

Stage 1: Tranmission of data

How do we transmit data?

The general idea here is to convert data bits to a square wave, where low is "0" and high is "1". The square wave is then modulated on top of a carrier. The modulated wave is then output to a underwater speaker.

For the construction of the square wave, we use the Agilent 3325A signal generator (SG), which is programmable using SCPI commands over its RS232 interface. Using the "DATA:DAC" command followed by a list of decimal integer values from -2047 to 2047, we can upload a custom waveform. These values correspond to the SG's 12-bit DAC codes, where -2047 represents the lowest voltage output and 2047 represents the highest voltage output. To represent bits in the square wave, we simply create a mapping between 0 and 1 to -2047 and 2047. Lets see an example.

The following commands results in a square wave with oscilating high and low, effcitevely representing the bits '10'.

DATA:DAC 2047 -2047 	(Download points to volatile memory)
DATA:COPY USER1 	(Copies the volatile memory to non-volatile under the name "USER1")
FUNC:USER USER1 	(Selects USER1 as the output USER wave)
FUNC USER 			(Outputs the USER wave)

The carrier frequency is simply a sine wave. On the SG the carrier is created using the the "APPLy" command.
The following a results in a sinus wave with frequency of 5000 Hz, peak to peak voltage of 3 and offset of -2.5 volt.

APPL:SIN 5.0E+3, 3.0, -2.5

We can now create both a square wave with out data encoded aswell as a carrier freq - we just need to combine the two. The signal generator supports AM, FM and FSK, and we use AM for not particular reason to combine the two (find a particular reason). We send the following commands to do AM:

AM:SOUR INT				(Select the source of modulation to be internal as you can also use an external source)
AM:INTernal:FUNCtion USER	(Select the currently selected USER function - this is our square wave)
AM:INT:FREQuency 1			(Set the frequency of the USER wave)
AM:DEPT 120					(Set modulation depths)
AM:STAT ON					(Output the modulated wave)

Until now we have only adjusted the voltage of the square wave, however, we have not yet determined its duration. If the square wave is too fast, it cannot be reconstructed on the reciever side. Thus, the transmitter and reciever should agree on a bit rate that they both can recieve and transmit. The SG can sample 200 MSa/s, where the reciever can only sample 96000 Sa/s, why the reciever limits the bit rate. 





duration should therefore not be be determined from the recievers cabaility of reading - which is its sample rate.


wave should match the sample rate of the reader which is 96k Hz: meaing 
As seen above the frequency is also specified. The frequency is how often the whole wave repeat itselfs. So a frequency of 1 Hz means that the whole wave repeats itself one time every second, where a frequency of 2 says that it repeats twice every second.

When used as the modulated wave, the list may only contain 8000 points.

 (SG) arbitrary wave functionality. The SG is programable using SCPI commands over the RS232 interface.DATA:DAC VOLATILE, 2047, -2047

own waves, which we will use to create the square wave. The creation of carrier wave is simple as it also allows to modulate on top of a carrier The signal generator supports simple modulation schemes such as AM, FM and FSK aswell as creating your own arbitrary waves.

To create an arbitrary wave you download points using the

SCPI commands sent over the RS232 protocol.

The transmitter side of system in Stage 1 consists of a underwater speaker (UW30) that is driven by a signal generator ).

To represent data bits we convert them to a square wave. The signal generator supports creating a user defined wave from 64k points.

The signal generator supports the creation arbitrary waveforms that can be downloaded using a remote interface.

The signal generator supports different standard waveforms (sinus, ramp, squares etc.) aswell as user defined ones. It also allows simple modulation schemes such as AM, FM and FSK.
For the transmission we create a carrier frequency

data across is to convert the data bits to a squarewave, that is then modulated on top of a carrier frequency.

The signal generator is cabable of creating different waveforms aswell as modulation schemes such as AM, FM and FSK.
