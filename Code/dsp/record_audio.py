import wave
import sys
import pyaudio
from config_values import RECORD_SECONDS, PATH_TO_WAV_FILE

CHUNK = 1024 # the amount of frames read per buffer, 1024 to balance between latency and processing load
# small chunk = reduces latency, but increases processing load
# large chunk = increases latency, but decreases processing load
FORMAT = pyaudio.paInt16
CHANNELS = 1 if sys.platform == 'darwin' else 2
RATE = 96000 # "hardcap" from StarTech

def create_wav_file_from_recording():
    p = pyaudio.PyAudio()

    # Open a new wave file
    wf = wave.open(PATH_TO_WAV_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)

    # # List available input devices
    # info = p.get_host_api_info_by_index(0)
    # numdevices = info.get('deviceCount')
    # for i in range(0, numdevices):
    #     if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
    #         print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

    # Open the audio stream
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print('Recording...')
    frames = []

    # Read and store audio data
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
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

if __name__ == "__main__":
    create_wav_file_from_recording()