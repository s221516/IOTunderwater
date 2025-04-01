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

def continuous_recording_with_preamble_detection():
    p = pyaudio.PyAudio()
    
    # Create a buffer to store the last few seconds of audio
    buffer_seconds = 5
    buffer_size = int(SAMPLE_RATE * buffer_seconds)
    audio_buffer = deque(maxlen=buffer_size)
    
    stream = p.open(format=FORMAT, 
                   channels=CHANNELS,
                   rate=SAMPLE_RATE,
                   input=True,
                   frames_per_buffer=CHUNK,
                   input_device_index=1)
    
    print('Starting continuous recording...')
    receiver = NonCoherentReceiver(BIT_RATE, CARRIER_FREQ)
    receiver.decode()

    try:
        while True:
            # Read audio data
            data = stream.read(CHUNK)
            audio_buffer.extend(np.frombuffer(data, dtype=np.int16))
            
            # Create temporary filename with timestamp
            temp_filename = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            
            # Save buffer to temporary file
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(np.array(list(audio_buffer)).tobytes())
            wf.close()
            
            try:
                receiver.decode()  # This will raise PreambleNotFoundError if no preamble is found
                
                # If we get here, a preamble was found
                print("Preamble detected! Saving audio...")
                
                # Record additional data
                additional_frames = []
                for _ in range(int(SAMPLE_RATE / CHUNK * 2)):
                    additional_frames.append(stream.read(CHUNK))
                
                # Save final file with additional data
                wf = wave.open(PATH_TO_WAV_FILE, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(np.array(list(audio_buffer)).tobytes())
                wf.writeframes(b''.join(additional_frames))
                wf.close()
                
                print("Audio saved to final location!")
                
            except PreambleNotFoundError:
                # No preamble found, delete temporary file
                os.remove(temp_filename)
                continue
            finally:
                # Clean up temporary file if it still exists
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                
    except KeyboardInterrupt:
        print("Stopping recording...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()