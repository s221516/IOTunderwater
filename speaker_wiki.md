
### Speaker

The speaker is a UW30 Underwater Loudspeaker.
It has the capacity to uniformly send a consistent signal in a **30 by 30** feet area pool.
___
###### Engineers specifications for the speaker UW30
- **Frequency Response** : $100 Hz$ to $10 kHz$
- **Power Handling** : 30 $Watts$
- **Impedance** : 8 $\Omega$

___

**Impedance** : is the concept of resistance in an electrical circuit. It is the measure of the opposition that a circuit presents to the flow of current when a voltage is applied. The symbol for impedance is $Z$. The unit of impedance is the ohm.

**Signal Attenuation** is when the signal is reduced in strength as it travels through the water.
  - Hence why we expect to need an amplifier to boost the received signal. 

Following is a list of formulas used to calculate the **signal attenuation** with respect to optical and electromagnetic singal attenuation.

___

**Electromagnetic**
Electromagnetic waves and radio frequency signals do not propagate well
underwater. This is mainly due to the high conductivity of
seawater. The penetration depth of electromagnetic waves is
inversely proportional both to the conductivity.
$(\sigma [S/m])$ and the frequency of the signal. 
$$\delta = \frac{1}{\sqrt{\pi f \mu \sigma}}$$
 **$f$** is the frequency of the signal, **$\mu$** is the permeability of the medium, and **$\sigma$** is the conductivity of the medium. $\delta [m]$ is defined as the distance that an electromagnetic wave travels before becoming attenuated by a factor of $1/e$ of its initial ampltitude (strength). 

 Signal with a small channel capacity can travel long distances in seawater, before they fade out. Therefore, the
antenna size (L [m]) increases, as the frequency decreases [45]:
$$L \approx \frac{v}{f}$$ where **$v$** is the speed of the electromagnetic wave in water, and **$f$** is the frequency of the signal.
___

###### Acoustic Signal Attenuation
Most underwater communication is done with acoustic links. However, the bandwidth of the acoustic signals is limited to that of in this case the speaker. In general the attenuation of acoustic signals is low for low frequencies, and increases with frequency. The attenuation of acoustic signals in seawater is given by the following formula: 
$$\alpha \approx F_{1}(f, pH) + F_{2}(f, T, S, z)$$ 
where $f$ is the frequency in [kHz], $pH$ is the acidity of the water, $T$ is the temperature, $S$ is the salinity, and $z$ is the depth of the water (kilometer). $F_{1}$ and $F_{2}$ are some unknown functions.

In our case we should make assumptions about the attenuation if deemed necessary. Important to realize is how it scales with frequency and how this is inversely proportional to the amount of information that can be transmitted by shannon formula.
___
### What is an optimal signal? 
In communication systems a signal can contain a theoritical maximum amount of information before there is a high probability of error. This is known as the **channel capacity**. The channel capacity is the maximum rate at which information can be transmitted over a communication channel. 
$$ C = B \log_2(1 + SNR)$$
**$C$** is the channel capacity in bits per second (bps),
where **$B$** is the bandwidth of the channel $(Hz)$, **$SNR$** is the signal-to-noise ratio, which measures the stregth of the signal relative to the noise.

`small example from deepseek`: 
If you have a channel with a bandwidth of 1 MHz and an SNR of 30 dB, the channel capacity can be calculated as follows:

Convert the SNR from decibels to a linear scale:
    $SNR_{linear} = 10^{30/10} = 1000$

###### Apply Shannon's formula:
$C \approx 1×10^6×log_2(1+1000)≈1×10^6×9.97≈9.97 Mbps$
This means the maximum data rate you can achieve on this channel, with negligible errors, is approximately 9.97 megabits per second.

In summary, channel capacity is a theoretical limit that defines the maximum amount of information that can be transmitted over a communication channel under given constraints, such as bandwidth and noise. It is a crucial concept for designing and evaluating communication systems.

___

#### Conclusion

The speaker optimally works in the frequency domain of $100 Hz$ to $10 kHz$. As we will for this project continue with accoustic signals, we will need to consider the attenuation of the signal in the water by an appoximation. We will also need to consider the channel capacity of the signal to determine the optimal signal to send. Furthermore, it is clear that we will need an amplifier for the speaker to work optimally.

___

#### Appendix

###### Calculations to justify the need for an amplifier
Speaker components:
- **Power Handling** : 30 $Watts$
- **Impedance** : 8 $\Omega$

Waveform generator components:
- **Voltage peak** : 42 V
- **Impedance** : 50 $\Omega$


