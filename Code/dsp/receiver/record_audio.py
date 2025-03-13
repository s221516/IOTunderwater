import wave
import sys
import pyaudio
from config_values import RECORD_SECONDS, PATH_TO_WAV_FILE, SAMPLE_RATE


#TODO FIX THIS BULLSHIT

CHUNK = 1024 # the amount of frames read per buffer, 1024 to balance between latency and processing load
# small chunk = reduces latency, but increases processing load
# large chunk = increases latency, but decreases processing load
FORMAT = pyaudio.paInt16
CHANNELS = 1 # this is either mono or stereo // mono = 1, stereo = 2, we do mono

def create_wav_file_from_recording(record_seconds=RECORD_SECONDS):
    p = pyaudio.PyAudio()

    # Open a new wave file
    wf = wave.open(PATH_TO_WAV_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    
    # List available input devices
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    # matches over all input devices in your computer, and prints them
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        device_name = device_info.get('name')
        print(f"DEVICE {device_name} {i}")
        


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