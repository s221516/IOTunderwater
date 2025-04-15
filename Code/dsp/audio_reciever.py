import wave
import pyaudio
import numpy as np
import os
import threading
from datetime import datetime
from collections import deque
from main import process_signal_for_chat
import config
from receiver.record_audio import create_wav_file_from_recording

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
        super().__init__()
        self.threshold = 400
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

    def record(self, p, pre_buffer, audio_chunk):
        recording_start = datetime.now()
        recording_buffer = list(pre_buffer)
        recording_buffer.extend(audio_chunk)
        if (datetime.now() - recording_start).total_seconds() >= RECORD_TIME:
            save_file = "chatting_recording.wav"
            save_path = os.path.join(SAVE_DIR, save_file)
            os.makedirs(SAVE_DIR, exist_ok=True)

            wf = wave.open(save_path, "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(np.array(recording_buffer).tobytes())
            wf.close()

    def calculate_rms(self, audio_chunk):
        """Calculate Root Mean Square of audio chunk"""
        chunk_float = audio_chunk.astype(np.float32)
        return np.sqrt(np.mean(np.square(chunk_float)))

    def monitor_audio(self):
        """Monitor audio input and record when threshold is exceeded"""
        p = pyaudio.PyAudio()
        new_recording = False
        # Moving average for smoothing audio levels
        rms_values = deque(maxlen=WINDOW_SIZE)

        # Create buffers
        pre_buffer_size = int(SAMPLE_RATE * PRE_RECORD_TIME)
        pre_buffer = deque(maxlen=pre_buffer_size)

        # State variables

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=INDEX_FOR_MIC,
        )
        try:

            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)

                # Calculate audio levels
                current_rms = self.calculate_rms(audio_chunk)
                rms_values.append(current_rms)
                avg_rms = np.mean(rms_values)
                # print(f"Avg RMS: {avg_rms}")
                # print(f"in transmit: {self.shared_state['is_transmitting']}")

                # Always keep recent audio in pre-buffer
                pre_buffer.extend(audio_chunk)

                # print the values of the boolean
                # print(f"Avg RMS: {avg_rms}")
                # print(f"Threshold: {self.threshold}")

                if avg_rms > self.threshold and (
                    not self.shared_state["is_transmitting"]
                ):
                    print("Debug: Started new recording")
                    print(f"Avg RMS: {avg_rms}")

                    # Start recording
                    create_wav_file_from_recording(RECORD_TIME)

                    msg = process_signal_for_chat(
                        config.CARRIER_FREQ,
                        config.BIT_RATE,
                    )
                    print("----------------")
                    print(msg)
                    print("----------------")
                    rms_values.clear()

        except KeyboardInterrupt:
            print("\nRecording stopped by user")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def run(self):
        self.monitor_audio()
