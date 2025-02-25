## Notes 
The underwater backscatter channel is a communication channel that uses the reflection of an acoustic signal off of a target to transmit information.

This is done by the new technology of piezo-acoustic backscatter (how does it work? I don't know, point is if it can be applied the cost of power for an underwater IoT unit is in the order or 5 to 6 times less than the current technologies).

##### Analytical model - to further understand and end-to-end underwater backscatter system

The paper manages to construct a numerical model in COMSOM Multiphysics and later test it in the real-world in a river. **Conclusion**: The model only deviates with a median of 0.71 dB from the real-world measurements..


![Underwater BackScatter architecture](image/underwater_backscatter/canoncal_underwater_backscatter_architecture.png)

#### Link Budgets
A link budget is an accounting of all of the gains and losses from the transmitter to the receiver in a communication system. Link budgets describe one direction of the wireless link. Most communications systems are bidirectional, so there must be a separate uplink and downlink budget. The “result” of the link budget will tell you roughly how much signal-to-noise ratio (abbreviated as SNR, which this textbook uses, or S/N) you should expect to have at your receiver. Further analysis would be needed to check if that SNR is high enough for your application.

You study link budgets not for the purpose of being able to actually make a link budget for some situation, but to learn and develop a system-layer point of view of wireless communications. [Click here for more information](https://pysdr.org/content/link_budgets.html?highlight=downlink)

