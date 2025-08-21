
import os
import wave
import pyaudio
import numpy as np
from collections import deque
import config
from datetime import datetime
from scipy.io import wavfile


class Recorder:
    def __init__(self):
        self.CHUNK = 1024   # the amount of frames read per buffer, 1024 to balance between latency and processing load
                            # small chunk = reduces latency, but increases processing load
                            # large chunk = increases latency, but decreases processing load
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1  # this is either mono or stereo // mono = 1, stereo = 2, we do mono
        self.LAST_PRINT_TIME = datetime.now()

        self.p = pyaudio.PyAudio()

    def calculate_recording_time(self, message):
        #TODO what if hamming encoding is used?
        len_of_bits = len(message) * 8 + 13  #13 for the barker preamble
        if config.USE_ESP:
            transmission_time = round((len_of_bits / config.BIT_RATE) * config.REP_ESP)
        else:
            transmission_time = round((len_of_bits / config.BIT_RATE) * config.REP_ESP)

        if transmission_time < 1:
            transmission_time = 1
            
        return transmission_time
    
    def listAvaliableDevices(self):
        # List available input devices
        info = self.p.get_host_api_info_by_index(0)
        numdevices = info.get("deviceCount")

        # matches over all input devices in your computer, and prints them
        for i in range(0, numdevices):
            device_info = self.p.get_device_info_by_host_api_device_index(0, i)
            device_name = device_info.get("name")
            print(f"DEVICE {device_name} {i}")

    def record(self, record_seconds, name):
        
        self.listAvaliableDevices()
        print(f"Recording for: {record_seconds} seconds")


        
        # Open a new wave file
        wf = wave.open("Code/dsp/data/raw_data/" + name + ".wav", "wb")
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(config.SAMPLE_RATE)

        # Open the audio stream
        stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=config.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            input_device_index=config.MIC_INDEX,
        )

        # print("Recording...")
        frames = []

        # Read and store audio data
        for _ in range(0, int(config.SAMPLE_RATE / self.CHUNK * record_seconds)):
            data = stream.read(self.CHUNK)
            frames.append(data)
        # print("Done recording")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        self.p.terminate()

        # Write the audio data to the wave file
        wf.writeframes(b"".join(frames))
        wf.close()
        print("Recording done")