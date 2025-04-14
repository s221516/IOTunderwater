import threading
import queue
import time
import wave
import pyaudio
import numpy as np
import os
from datetime import datetime
from collections import deque

import config
from receiver.receiverClass import NonCoherentReceiver
from errors import PreambleNotFoundError

# Audio recording constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = 1024
SAMPLE_RATE = config.SAMPLE_RATE
SAVE_DIR = "Code/dsp/data"
THRESHOLD = 3500
RECORD_TIME = 5.0
PRE_RECORD_TIME = 0.5
WINDOW_SIZE = 100


class InteractiveTransceiver:
    def __init__(self, use_esp=True):
        self.use_esp = use_esp
        self.recording_active = threading.Event()
        self.stop_requested = threading.Event()
        self.audio_queue = queue.Queue()
        self.message_queue = queue.Queue()

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

        # Status display control
        self.status_line = ""
        self.input_active = False
        self.input_lock = threading.Lock()

    def update_status_line(self, text):
        """Update the status line without interfering with input"""
        with self.input_lock:
            if self.input_active:
                return
        if self.status_line:
            # Move cursor up one line and clear it
            print("\033[F\033[K", end="")
        print(text)
        self.status_line = text

    def calculate_rms(self, audio_chunk):
        """Calculate Root Mean Square of audio chunk"""
        chunk_float = audio_chunk.astype(np.float32)
        return np.sqrt(np.mean(np.square(chunk_float)))

    def update_status_line(self, text):
        """Update the status line without interfering with input"""
        if self.status_line:
            # Move cursor up one line and clear it
            print("\033[F\033[K", end="")
        print(text)
        self.status_line = text

    def recording_thread(self):
        """Thread that continuously records audio and detects signals"""
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

            print("Recording thread started - listening for signals...")
            print("Audio levels will be shown below:")
            print("")  # Initial status line placeholder
            self.status_line = ""

            last_print_time = datetime.now()
            PRINT_INTERVAL = 0.5  # Print audio levels every half second

            while not self.stop_requested.is_set():
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16)

                    # Calculate audio levels
                    current_rms = self.calculate_rms(audio_chunk)
                    self.rms_values.append(current_rms)
                    avg_rms = np.mean(self.rms_values)

                    # Print audio level periodically
                    current_time = datetime.now()
                    if (
                        current_time - last_print_time
                    ).total_seconds() >= PRINT_INTERVAL:
                        self.update_status_line(
                            f"Audio Level: {avg_rms:8.2f} | {'*' * int(avg_rms/100)}"
                        )
                        last_print_time = current_time

                    # Always keep recent audio in pre-buffer
                    self.pre_buffer.extend(audio_chunk)

                    # If not already recording and audio level exceeds threshold
                    if not self.is_recording and avg_rms > THRESHOLD:
                        self.update_status_line("Sound detected! Recording...")
                        self.is_recording = True
                        self.recording_start = datetime.now()
                        self.recording_buffer = list(self.pre_buffer)
                        self.recording_active.set()

                    # If currently recording
                    if self.is_recording:
                        self.recording_buffer.extend(audio_chunk)

                        # Check if recording time has elapsed
                        if (
                            datetime.now() - self.recording_start
                        ).total_seconds() >= RECORD_TIME:
                            self.update_status_line("Recording complete, processing...")

                            # Save the recording
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            save_file = f"recording_{timestamp}.wav"
                            save_path = os.path.join(SAVE_DIR, save_file)

                            # Ensure the directory exists
                            os.makedirs(SAVE_DIR, exist_ok=True)

                            wf = wave.open(save_path, "wb")
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(self.p.get_sample_size(FORMAT))
                            wf.setframerate(SAMPLE_RATE)
                            wf.writeframes(np.array(self.recording_buffer).tobytes())
                            wf.close()

                            self.update_status_line(f"Recording saved as: {save_path}")

                            # Process the recording
                            self.process_recording(save_path)

                            # Reset recording state
                            self.is_recording = False
                            self.recording_buffer = []
                            self.recording_active.clear()

                            # Reset the status line after processing
                            self.update_status_line(
                                f"Audio Level: {avg_rms:8.2f} | {'*' * int(avg_rms/100)}"
                            )

                except IOError:
                    # Handle potential buffer overflow errors
                    continue

        except Exception as e:
            print(f"\nError in recording thread: {e}")
        finally:
            if "stream" in locals():
                stream.stop_stream()
                stream.close()
            print("Recording thread stopped")

    def process_recording(self, recording_path):
        """Process a recorded audio file"""
        try:
            # Update config path for the decoder
            config.PATH_TO_WAV_FILE = recording_path

            # Create non-coherent receivers
            receiver = NonCoherentReceiver(
                config.BIT_RATE, config.CARRIER_FREQ, band_pass=False
            )
            receiver_bandpass = NonCoherentReceiver(
                config.BIT_RATE, config.CARRIER_FREQ, band_pass=True
            )

            # Decode the signal
            message, debug = receiver.decode()
            message_bp, debug_bp = receiver_bandpass.decode()

            print("\n--- DECODED MESSAGE ---")
            print(f"Without bandpass: {message}")
            print(f"With bandpass: {message_bp}")
            print("")  # Add a blank line after results

        except PreambleNotFoundError:
            print("\nNo preamble found in the recording")
        except Exception as e:
            print(f"\nError processing recording: {e}")

    def transmit_message(self, message):
        """Transmit a message using either ESP32 or signal generator"""
        try:
            if self.use_esp:
                import esp32test

                print(f"Transmitting '{message}' via ESP32...")
                esp32test.transmit_to_esp32(
                    message, config.CARRIER_FREQ, config.BIT_RATE
                )
            else:
                from transmitterPhysical import transmitPhysical, stopTransmission

                print(f"Transmitting '{message}' via signal generator...")
                transmitPhysical(message, config.CARRIER_FREQ, config.BIT_RATE)
                time.sleep(2)  # Allow time for transmission
                stopTransmission()

            print("Transmission complete")
            print("")  # Add a blank line after transmission
        except Exception as e:
            print(f"Transmission error: {e}")

    def input_thread(self):
        """Thread that handles user input"""
        print("\n--- INTERACTIVE TRANSCEIVER ---")
        print("Enter messages to transmit or commands:")
        print("  'exit' to quit")
        print("  'esp' to switch to ESP32 transmitter")
        print("  'sg' to switch to signal generator")
        print("  'cf=<freq>' to set carrier frequency (e.g., 'cf=6000')")
        print("  'br=<rate>' to set bit rate (e.g., 'br=100')")

        while not self.stop_requested.is_set():
            try:
                # Position the prompt consistently
                with self.input_lock:
                    self.input_active = True
                user_input = input("> ")
                with self.input_lock:
                    self.input_active = False

                if user_input.lower() == "exit":
                    self.stop_requested.set()
                    break

                elif user_input.lower() == "esp":
                    self.use_esp = True
                    print("Switched to ESP32 transmitter")

                elif user_input.lower() == "sg":
                    self.use_esp = False
                    print("Switched to signal generator")

                elif user_input.lower().startswith("cf="):
                    try:
                        freq = int(user_input[3:])
                        config.CARRIER_FREQ = freq
                        print(f"Carrier frequency set to {freq} Hz")
                    except ValueError:
                        print("Invalid frequency format. Use 'cf=6000' format.")

                elif user_input.lower().startswith("br="):
                    try:
                        rate = int(user_input[3:])
                        config.BIT_RATE = rate
                        print(f"Bit rate set to {rate} bps")
                    except ValueError:
                        print("Invalid bit rate format. Use 'br=100' format.")

                else:
                    # Don't transmit if currently recording
                    if self.recording_active.is_set():
                        print("Please wait, currently recording...")
                    else:
                        self.transmit_message(user_input)

            except EOFError:
                break

            except KeyboardInterrupt:
                self.stop_requested.set()
                break

    def start(self):
        """Start the interactive transceiver"""
        # Start recording thread
        rec_thread = threading.Thread(target=self.recording_thread)
        rec_thread.daemon = True
        rec_thread.start()

        # Start input thread
        input_thread = threading.Thread(target=self.input_thread)
        input_thread.daemon = True
        input_thread.start()

        try:
            # Wait for stop request
            while not self.stop_requested.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            print("\nShutting down...")
            self.stop_requested.set()
            input_thread.join(timeout=1)
            rec_thread.join(timeout=1)
            self.p.terminate()
            print("Done.")


if __name__ == "__main__":
    # Default to ESP32 transmission
    transceiver = InteractiveTransceiver(use_esp=True)
    transceiver.start()
