# Underwater Acoustic Communication System

An underwater digital signal processing (DSP) communication system built on the **ESP32** microcontroller. This project demonstrates acoustic data transmission through water using **Amplitude Modulation (AM)**, taking inspiration from well-established terrestrial wireless communication techniques and adapting them to the unique challenges of the underwater channel.

## Background

Terrestrial wireless communication relies on radio-frequency (RF) electromagnetic waves, but RF signals attenuate rapidly in water. Acoustic waves, on the other hand, propagate much more effectively underwater. This project applies core principles from terrestrial digital communication — carrier modulation, preamble synchronization, and forward error correction — to build a working underwater acoustic link between a transmitter (underwater speaker) and a receiver (hydrophone).

## Features

- **Amplitude Modulation (AM)** of binary data onto a configurable carrier frequency (default 11 kHz)
- **Barker-13 preamble** for reliable frame synchronization and start-of-message detection
- **Non-coherent envelope detection** on the receiver side
- **Optional forward error correction** using Hamming (7,4) codes or convolutional coding with Viterbi decoding
- **Configurable bit rate** (tested from 25 bps up to 2000 bps) and carrier frequency (1–25 kHz)
- **ESP32 transmitter firmware** using I2S for audio output, or signal generator (Agilent 33250A) via SCPI commands
- **Python-based DSP receiver** with real-time recording, demodulation, and visualization
- **Extensive test data** from pool and tank experiments at various distances (1 m, 3 m, 5 m, 6 m)

## Project Structure

```
├── Code/
│   ├── dsp/                        # Python DSP processing
│   │   ├── main.py                 # Entry point – transmit & receive pipeline
│   │   ├── config.py               # Global parameters (bit rate, carrier freq, etc.)
│   │   ├── Transmitter.py          # Transmitter control (signal generator or ESP32)
│   │   ├── receiver/               # Receiver, recording, and demodulation
│   │   ├── encoding/               # Hamming & convolutional encoding/decoding
│   │   ├── visuals/                # Signal visualization utilities
│   │   └── data/                   # Recorded .wav files and test results
│   └── ESP32_code/                 # ESP32 firmware (C, ESP-IDF)
│       ├── main/
│       │   ├── main.c              # Transmitter firmware (AM modulation via I2S)
│       │   └── i2s.c               # I2S audio driver
│       └── CMakeLists.txt
├── Equipment/                      # Hardware datasheets (hydrophone, speaker)
├── Wikis/                          # Technical documentation & research notes
├── requirements.txt                # Python dependencies
└── sdkconfig                       # ESP-IDF board configuration
```

## Prerequisites

### Hardware

- **ESP32 development board** (e.g. ESP32-S3) — or an Agilent 33250A signal generator
- **Underwater speaker** for acoustic transmission
- **Hydrophone** (e.g. Aquarian Audio AS-1) for reception
- **Sound card** supporting a sample rate of at least 96 kHz
- USB cables for serial communication

### Software

- **Python 3.7+**
- **ESP-IDF** (Espressif IoT Development Framework) — required only for building the ESP32 firmware
- **Git**

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/s221516/IOTunderwater.git
cd IOTunderwater
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `pyaudio` requires the PortAudio system library. On macOS install it with `brew install portaudio`; on Ubuntu/Debian use `sudo apt-get install portaudio19-dev`.

### 3. Configure the DSP Parameters

Edit `Code/dsp/config.py` to match your setup:

| Parameter | Default | Description |
|---|---|---|
| `TRANSMITTER_PORT` | `"COM11"` | Serial port of the transmitter device |
| `USE_ESP` | `False` | Set to `True` when using an ESP32 instead of a signal generator |
| `MIC_INDEX` | `2` | Audio input device index for the hydrophone/microphone |
| `BIT_RATE` | `500` | Transmission bit rate in bps |
| `CARRIER_FREQ` | `11000` | Carrier frequency in Hz |
| `SAMPLE_RATE` | `96000` | Audio sample rate (fixed by sound card) |
| `HAMMING_CODING` | `False` | Enable Hamming (7,4) error correction |
| `CONVOLUTIONAL_CODING` | `False` | Enable convolutional coding with Viterbi decoding |

### 4. Build & Flash the ESP32 Firmware (Optional)

If you are using an ESP32 as the transmitter:

```bash
cd Code/ESP32_code

# Set your target chip
idf.py set-target esp32s3

# Build, flash, and open the serial monitor
idf.py build
idf.py flash
idf.py monitor
```

## Running the Project

Once the hardware is connected and the configuration is set:

```bash
cd Code/dsp
python main.py
```

This will:

1. **Encode** the message into a binary bit stream and prepend the Barker-13 preamble.
2. **Modulate** the bit stream onto the carrier frequency using AM.
3. **Transmit** the signal through the underwater speaker (via the signal generator or ESP32).
4. **Record** the received acoustic signal through the hydrophone.
5. **Demodulate** the recording using non-coherent envelope detection.
6. **Decode** the payload, applying error correction if enabled.
7. **Log** results (Hamming distance, signal power, metadata) to a CSV file and optionally display signal visualizations.

## License

This project is developed for academic and research purposes.
