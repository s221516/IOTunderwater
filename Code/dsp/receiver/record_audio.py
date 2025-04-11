import wave
import pyaudio
import numpy as np
import os
from collections import deque
from config import PATH_TO_WAV_FILE, SAMPLE_RATE, BIT_RATE, CARRIER_FREQ
from datetime import datetime
from receiver.receiverClass import NonCoherentReceiver
from errors import PreambleNotFoundError

CHUNK = 1024 # the amount of frames read per buffer, 1024 to balance between latency and processing load
# small chunk = reduces latency, but increases processing load
# large chunk = increases latency, but decreases processing load
FORMAT = pyaudio.paInt16
CHANNELS = 1 # this is either mono or stereo // mono = 1, stereo = 2, we do mono

def create_wav_file_from_recording(record_seconds):
    p = pyaudio.PyAudio()

    # Open a new wave file
    wf = wave.open(PATH_TO_WAV_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    
    # # List available input devices
    # info = p.get_host_api_info_by_index(0)
    # numdevices = info.get('deviceCount')

    # # matches over all input devices in your computer, and prints them
    # for i in range(0, numdevices):
    #     device_info = p.get_device_info_by_host_api_device_index(0, i)
    #     device_name = device_info.get('name')
    #     print(f"DEVICE {device_name} {i}")

    # Open the audio stream
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, 
                    frames_per_buffer=CHUNK, input_device_index=1)

    print('Recording...')
    frames = []

    # Read and store audio data
    for _ in range(0, int(SAMPLE_RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
    print('Done recording')

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Write the audio data to the wave file
    wf.writeframes(b''.join(frames))
    wf.close()

def calculate_rms(audio_chunk):
    """Calculate Root Mean Square of audio chunk"""
    # Convert to float for better precision
    chunk_float = audio_chunk.astype(np.float32)
    # Calculate RMS
    return np.sqrt(np.mean(np.square(chunk_float)))

def continuous_recording_with_threshold():
    p = pyaudio.PyAudio()
    
    # Add these constants at the start of the function
    SAVE_DIR = "Code/dsp/data"
    SAVE_FILE = "recording_test.wav"

    # Parameters
    THRESHOLD = 1000  
    RECORD_TIME = 7.0  # Record for exactly 7 seconds
    PRE_RECORD_TIME = 0.5
    PRINT_INTERVAL = 1.0
    
    # Moving average for smoothing audio levels
    WINDOW_SIZE = 10
    rms_values = deque(maxlen=WINDOW_SIZE)
    
    # Create buffers
    pre_buffer_size = int(SAMPLE_RATE * PRE_RECORD_TIME)
    pre_buffer = deque(maxlen=pre_buffer_size)
    recording_buffer = []
    
    # State variables
    is_recording = False
    recording_start = None
    last_print_time = datetime.now()
    
    stream = p.open(format=FORMAT, 
                   channels=CHANNELS,
                   rate=SAMPLE_RATE,
                   input=True,
                   frames_per_buffer=CHUNK,
                   input_device_index=1)
    
    print("Listening for audio above threshold...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            data = stream.read(CHUNK)
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            
            current_rms = calculate_rms(audio_chunk)
            rms_values.append(current_rms)
            avg_rms = np.mean(rms_values)
            
            # Print audio level once per second
            current_time = datetime.now()
            if (current_time - last_print_time).total_seconds() >= PRINT_INTERVAL:
                print(f"\rAudio Level (RMS): {avg_rms:8.2f} | {'*' * int(avg_rms/100)}", end='', flush=True)
                last_print_time = current_time
            
            # Always keep recent audio in pre-buffer
            pre_buffer.extend(audio_chunk)
            
            if not is_recording and avg_rms > THRESHOLD:
                print("\nSound detected! Recording for 7 seconds...")
                is_recording = True
                recording_start = datetime.now()
                recording_buffer = list(pre_buffer)
                
            if is_recording:
                recording_buffer.extend(audio_chunk)
                
                # Check if 7 seconds have elapsed
                if (datetime.now() - recording_start).total_seconds() >= RECORD_TIME:
                    print("\nRecording complete, saving...")
                    
                    # Save the recording
                    save_path = os.path.join(SAVE_DIR, SAVE_FILE)
                    wf = wave.open(save_path, 'wb')
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(p.get_sample_size(FORMAT))
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(np.array(recording_buffer).tobytes())
                    wf.close()
                    
                    print(f"Recording saved as: {save_path}")
                    
                    is_recording = False
                    recording_buffer = []
            
    except KeyboardInterrupt:
        print("\nRecording stopped by user")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    continuous_recording_with_threshold()