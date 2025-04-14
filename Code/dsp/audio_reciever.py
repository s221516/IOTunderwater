import threading
import time
import wave
import pyaudio
import numpy as np
import os
from datetime import datetime
from collections import deque
from main import main

import config
from receiver.receiverClass import NonCoherentReceiver
from errors import PreambleNotFoundError

# Audio recording constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = 1024
SAMPLE_RATE = config.SAMPLE_RATE
SAVE_DIR = "Code/dsp/data"
THRESHOLD = 1000
RECORD_TIME = 5.0
PRE_RECORD_TIME = 0.5
WINDOW_SIZE = 100


class AudioReceiver:
    def __init__(self):
        """Initialize the audio receiver for continuous monitoring"""
        self.stop_requested = threading.Event()

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()

        # List available input devices
        print("\nAvailable audio input devices:")
        info = self.p.get_host_api_info_by_index(0)
        numdevices = info.get("deviceCount")

        for i in range(0, numdevices):
            device_info = self.p.get_device_info_by_host_api_device_index(0, i)
            device_name = device_info.get("name")
            print(f"DEVICE {device_name} {i}")

        # Allow for device selection
        try:
            self.device_index = int(
                input("Select input device index (or press Enter for default): ")
                or "-1"
            )
            if self.device_index < 0:
                self.device_index = None
                print("Using default input device")
            else:
                print(f"Using device: {self.device_index}")
        except ValueError:
            self.device_index = None
            print("Invalid input. Using default input device")

        # Moving average for smoothing audio levels
        self.rms_values = deque(maxlen=WINDOW_SIZE)

        # Create pre-buffer
        self.pre_buffer_size = int(SAMPLE_RATE * PRE_RECORD_TIME)
        self.pre_buffer = deque(maxlen=self.pre_buffer_size)

        # Recording state
        self.is_recording = False
        self.recording_buffer = []
        self.recording_start = None

    def calculate_rms(self, audio_chunk):
        """Calculate Root Mean Square of audio chunk"""
        chunk_float = audio_chunk.astype(np.float32)
        return np.sqrt(np.mean(np.square(chunk_float)))

    def start_monitoring(self):
        """Start monitoring audio input"""
        # Set up stream with or without specific device index
        stream_args = {
            "format": FORMAT,
            "channels": CHANNELS,
            "rate": SAMPLE_RATE,
            "input": True,
            "frames_per_buffer": CHUNK,
        }

        # Only add device index if specified
        if self.device_index is not None:
            stream_args["input_device_index"] = self.device_index

        try:
            stream = self.p.open(**stream_args)

            print("\n--- AUDIO MONITORING ACTIVE ---")
            print("Listening for signals above threshold...")
            print("Press Ctrl+C to stop")

            last_print_time = datetime.now()
            print_interval = 1.0  # Print audio levels every second

            while not self.stop_requested.is_set():
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16)

                    # Calculate audio levels
                    current_rms = self.calculate_rms(audio_chunk)
                    self.rms_values.append(current_rms)
                    avg_rms = np.mean(self.rms_values)

                    # # Print audio level periodically
                    # current_time = datetime.now()
                    # if (
                    #     current_time - last_print_time
                    # ).total_seconds() >= print_interval:
                    #     print(
                    #         f"\rAudio Level: {avg_rms:8.2f} | {'*' * int(avg_rms/100)}",
                    #         end="",
                    #     )
                    #     last_print_time = current_time

                    # Always keep recent audio in pre-buffer
                    self.pre_buffer.extend(audio_chunk)

                    # If not already recording and audio level exceeds threshold
                    if not self.is_recording and avg_rms > THRESHOLD:
                        print("\nSound detected! Recording...")
                        self.is_recording = True
                        self.recording_start = datetime.now()
                        self.recording_buffer = list(self.pre_buffer)

                    # If currently recording
                    if self.is_recording:
                        self.recording_buffer.extend(audio_chunk)

                        # Check if recording time has elapsed
                        if (
                            datetime.now() - self.recording_start
                        ).total_seconds() >= RECORD_TIME:
                            print("\nRecording complete, processing...")

                            # Save the recording
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            save_file = f"recording_test.wav"
                            save_path = os.path.join(SAVE_DIR, save_file)

                            # Ensure the directory exists
                            os.makedirs(SAVE_DIR, exist_ok=True)

                            wf = wave.open(save_path, "wb")
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(self.p.get_sample_size(FORMAT))
                            wf.setframerate(SAMPLE_RATE)
                            wf.writeframes(np.array(self.recording_buffer).tobytes())
                            wf.close()

                            print(f"Recording saved as: {save_path}")

                            main()
                            # Reset recording state
                            self.is_recording = False
                            self.recording_buffer = []

                except IOError:
                    # Handle potential buffer overflow errors
                    continue

        except Exception as e:
            print(f"\nError in audio monitoring: {e}")
        finally:
            if "stream" in locals():
                stream.stop_stream()
                stream.close()
            print("Audio monitoring stopped")


if __name__ == "__main__":
    receiver = AudioReceiver()
    try:
        receiver.start_monitoring()
    except KeyboardInterrupt:
        print("\nStopping audio monitoring...")
    finally:
        receiver.stop()
        receiver.p.terminate()
