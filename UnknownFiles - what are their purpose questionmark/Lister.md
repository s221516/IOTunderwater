Description

The aim of this project is to design and implement a basic underwater communication system that can transmit data from an \textbf{underwater speaker} to a \textbf{hydrophone receiver}. This will be achieved by generating and transmitting a \textbf{carrier frequency} through a \textbf{signal generator} connected to the underwater speaker, which will be received by a hydrophone.

The project involves both the transmission and reception of a carrier frequency and the \textbf{modulation of this signal to encode binary data for transfer.} The project includes fundamental hardware setup, including generating the carrier signal at a specified frequency and monitoring the transmitted and received signals using an \textbf{oscilloscope}.  To modulate the carrier frequency and encode data, techniques such as \textbf{amplitude, frequency, or phase modulation} may be considered to represent binary data. \textbf{Encoding schemes} should be also considered to ensure that the information can be reliably decoded upon reception.

Integrating the physical and data link layers is important in this project to automate data transfer and enable analysis processes.

Things to do the 21/02/2025

- Understand Hilbert (Morten)
- Make notebook of am_communication (Mathias)
- Make the am_comm work with an .wav file
- Make a project plan!! where we can also move this stupid lists to

Questions for Haris

- Which way should we go? Hardcore signal processing or do IoUT?
- Can you send other project plans + earlier bachelor projects
- What are the expectations for the lab meeting in march?

To Do:

- Investigate easy programable solution to output all kind of waves - test with arduino uno maybe (Felipe).
- Investigate how to connect to signal generator and osciloscope to computer - ask for cables
- meld til kursus
- Fix github, så Lister ikke findes

Notes with Mathias' friend:

- Bodeplot
- Lowpass filter -> active filter and passive filter (first order to foruth order)
- kapasitor
- figure out micro-controller with the highest sample rate

StepX:

- How to modulate a carrier wave?

Step2: (mostly understood)

- Geneartor - what dooes the generator actually output? Is it a match?
- Speaker - what does 8 ohm and 30 Watt means. And how does it relate to signal generator.
- In general what should work? Big paper

To buy:

- (ekstra amplifier)
- rs-232 to usb (Haris is checking, else -> https://shorturl.at/0b44g - maybe 2x, s.t. wavegenerator -> computer and oscilloscope -> computer ), possible a null modem adapter

Finished steps
--------------

Step 1: Base case test Done 2/5:


- test0: Osciloscope + wave generator *check()
- test1: Osciloscope + med hydrophone *check()
- test2: Speaker + wave generator (spil sinus waves)
- base test: Wave generator + speaker -> hydrophone + Osciloscope
