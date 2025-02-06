- [ ] `Underwater Speaker` 
	- [ ] See videos about general speakers. 
- [x] `Hydrophone Receiver` 
- [ ] `Carrier Frequency` 
- [ ] `signal generator` 
	- [ ] Device to construct the digital modulation
- [ ] `modulation of this signal to encode binary data for transfer` 
	- [ ] see [[Spring 2025/Bachelor Project gits/attachments/Pasted image 20250203145413.png| modulation examples]]
- [ ] `Oscilloscope`
	- [ ] Device to monitor signal functions visually
- [ ] `amplitude, frequency, or phase modulation`
- [ ] `Encoding Scheme` 
	- [ ] Is there a standard set i.e. ASCII encoding scheme we have to use or do we make use of an encoding scheme that we make ourselves?? based on the previous techniques mentioned.

##### How does sound work
in general the speed of sound varies in different materials: 
[How does sound travel](https://www.youtube.com/watch?v=1kjAkuwYx2M) 
[How do speakers work?](https://www.youtube.com/watch?v=jhg90zsjqt4)
see timestamp at 3min 

![[Spring 2025/Bachelor Project gits/attachments/Pasted image 20250203154819.png]]

`Modulation`: Modulation is the concept of  shaping the carrier frequency into a form that gives the receiver information about the data being transmitted through wave, sound light etc. 
Wave function to calculate sound pressure in 
$$\Delta P (x,t)= \Delta P_{max} \cdot sin(kx \mp \omega t + \phi)$$
where: 
- $\Delta P_{max}$ : the max pressure
- $k = \frac{2 \pi}{\lambda}$ is the wave number
- $\omega = \frac{2 \pi}{T} = 2 \pi f$ is the angular frequency
- $\phi$ is the initial phase
- x is displacement
- t is time 



- Amplitude : max |height| of the wave. 
- Frequency : $s^{-1}$ , cycles per second 
- Phase : the value of the wave function when x and t is set to 0

##### Modulation
there are dofferent modulation schemes i.e. 
<img src="attachments/modulation types.png" width="500" height="500">
we somehow have to reliably transmit and decode a digital modulation.

##### Material

- [Waveform Generator Datasheet](Data-sheets/Waveform-generator-datasheet.pdf)
-  [Amplifier Xli 3500 series](Data-sheets/Amplifiers(XLi_800__XLi1500__XLi2500__XLi3500)_DoC_7-7-14.pdf)
-  

- [IoUT mega paper](Data-sheets/Internet%20of%20Underwater%20Things%20and%20Big%20Marine%20Data%20Analytics—A%20Comprehensive%20Survey.pdf)

- [underwater speaker](https://www.performanceaudio.com/products/electro-voice-uw30-underwater-loudspeaker)
- [Hydrophone](https://www.aquarianaudio.com/h2a-hydrophone.html)

#### NOISE?? 

##### Understand Digital signal processing better through python 
[**here**](https://pysdr.org/content/intro.html#purpose-and-target-audience)****

### excalidraw of project (very high level)
[[Spring 2025/Bachelor Project gits/drawings/drawing of initial understanding of bs project]]
