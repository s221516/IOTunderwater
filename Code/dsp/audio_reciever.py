import time
import wave
import pyaudio
import numpy as np
import os
import threading
from datetime import datetime
from collections import deque

from sympy import N
from main import process_signal_for_chat
import config

# from receiver.record_audio import create_wav_file_from_recording

# Audio recording constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = 1024
SAMPLE_RATE = config.SAMPLE_RATE
SAVE_DIR = "Code/dsp/data"
INDEX_FOR_MIC = 2  # Change this depending on your setup
RECORD_TIME = 5.0
PRE_RECORD_TIME = 0.5
WINDOW_SIZE = 100


class AudioReceiver(threading.Thread):
    def __init__(self, shared_state):
        super().__init__(name="AudioReceiverThread")
        self.threshold = 500
        self.shared_state = shared_state

    def get_is_new_recording(self):
        return self.new_recording

    def set_is_new_recording(self, flag):
        self.new_recording = flag

    def list_audio_devices(self):
        """List all available audio input devices"""
        p = pyaudio.PyAudio()
        print("\nAvailable audio input devices:")
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get("deviceCount")

        for i in range(0, numdevices):
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            device_name = device_info.get("name")
            print(f"DEVICE {device_name} {i}")
        p.terminate()

    def create_wav_file_from_recording(
        self, record_seconds, stream, sample_size_format
    ):

        # Open a new wave file
        wf = wave.open(config.PATH_TO_WAV_FILE, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_size_format)
        wf.setframerate(SAMPLE_RATE)

        print("Recording...")
        frames = []

        # Read and store audio data
        for _ in range(0, int(SAMPLE_RATE / CHUNK * record_seconds)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("Done recording")

        # Stop and close the stream

        # Write the audio data to the wave file
        wf.writeframes(b"".join(frames))
        wf.close()

    def calculate_rms(self, audio_chunk):
        """Calculate Root Mean Square of audio chunk"""
        chunk_float = audio_chunk.astype(np.float32)
        return np.sqrt(np.mean(np.square(chunk_float)))

    def monitor_audio(self):
        """Monitor audio input and record when threshold is exceeded"""
        p = pyaudio.PyAudio()
        # Moving average for smoothing audio levels
        rms_values = deque(maxlen=WINDOW_SIZE)

        # Create buffers
        pre_buffer_size = int(SAMPLE_RATE * PRE_RECORD_TIME)
        pre_buffer = deque(maxlen=pre_buffer_size)

        stream = None

        try:

            while True:

                stream = p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=INDEX_FOR_MIC,
                )

                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)

                # Calculate audio levels
                current_rms = self.calculate_rms(audio_chunk)
                avg_rms = np.mean(current_rms)
                # print(f"Avg RMS: {avg_rms}")
                # print(f"in transmit: {self.shared_state['is_transmitting']}")

                # Always keep recent audio in pre-buffer
                pre_buffer.extend(audio_chunk)

                # print the values of the boolean
                # print(f"Avg RMS: {avg_rms}")
                # print(f"Threshold: {self.threshold}")

                if current_rms > self.threshold and (
                    not self.shared_state["is_transmitting"]
                ):

                    print("Debug: Started new recording")
                    print(f"Avg RMS: {current_rms}")

                    if self.shared_state["msg"] is not None:

                        # print("Debug: Started new recording")
                        # print(f"Avg RMS: {current_rms}")
                        # Start recording
                        len_of_bits = len(self.shared_state["msg"]) * 8 + 13
                        record_time = config.REP_ESP * (len_of_bits / config.BIT_RATE)
                        self.create_wav_file_from_recording(
                            record_time, stream, p.get_sample_size(FORMAT)
                        )

                        msg, msg_bp = process_signal_for_chat(
                            config.CARRIER_FREQ,
                            config.BIT_RATE,
                        )
                        print("----------------")
                        print(f"Received             : {msg}")
                        print(f"Received w. band-pass: {msg_bp}")
                        print("----------------")

                    time.sleep(2)

                    # reset the pre-buffer after recording
                stream.stop_stream()
                stream.close()

        except KeyboardInterrupt:
            print("\nRecording stopped by user")
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
                p.terminate()

    def run(self):
        self.monitor_audio()
