Modulation

Stage 1: Tranmission of data

The general idea here is to convert data bits to a square wave, where low is 0 and high is 1. The square wave is then modulated on top of a carrier. The modulated wave is then output to a underwater speaker.

For the construction of the square wave, we use the Agilent 3325A function generator (FG, which programmable using SCPI commands over its RS232 interface. Using the "DATA:DAC" command an arbitrary waveform can be created from a list of decimal integer values from -2047 to 2047. These values correspond to the FG's 12-bit DAC codes, where -2047 represents the lowest voltage output and 2047 represents the highest.

The FG takes the list of specified points and expands them to fill waveform memory. If you download less than 16,384 (16K) points, a waveform with 16,384 points is automatically generated. However, when using the arbitrary waveform as the modulating wave, these points is decimated to only 8k (probably 8192, it is not mentioned specifaclly). In order to create a single bit, we must use two points. This means that we can at most transmit 4000 bits, since more would be lost in the decimation.

Here is an example program on how to create "10" as a square wave

DATA:DAC 2047, -2047  (The list [2047, -2047] is expanded up to 16k points, which results in a list of 16k entries: [2047, 2047 .... -2047, -2047] )
DATA:COPY USER1 	(Copies the volatile memory to non-volatile under the name "USER1")
FUNC:USER USER1 	(Selects USER1 as the output USER wave)
FUNC USER 			(Outputs the USER wave)

The points downloaded allways consists of a single period of the wave. The FG allows to adjust frequency to change the duration of the wave. However, what should our bit rate be? If the frequency is is too large the bits become too compact, such that the sample rate on the reciever side is not high enough to reconstruct it. To fix this the transmitter and reciever agrees on a bit rate that they both can recieve and transmit.


Lets do an example on how we calculate the bit rate. We have decided a bit rate, and we send a wave containing N bits. 

frequency = bitrate/(len(bits)).

This means if we agree on a bit rate of 2 and we send "10", the frequency is 1.
If the bit rate stays the same and we send "1010", the frequency is 2. 

For the carrier wave we use the the "APPLy" command.
The following a results in a output sinus wave with frequency of 5000 Hz, peak to peak voltage of 3 and offset of -2.5 volt.

APPL:SIN 5.0E+3, 3.0, -2.5

Now we need to modulate the square wave on top of the carrier. The signal generator supports AM, FM and FSK, and we simply use AM. 
We send the following commands to do AM:

AM:SOUR INT				(Select the source of modulation to be internal as you can also use an external source)
AM:INTernal:FUNCtion USER	(Select the currently selected USER function - this is our square wave)
AM:INT:FREQuency 1			(Set the frequency of the USER wave)
AM:DEPT 120					(Set modulation depths)
AM:STAT ON					(Output the modulated wave)



from page 296-297 (https://www.youtube.com/watch?v=iq76RbnAbDk even explain in youtube)

- write how DDS works
- how the frequency chaging with regards to
