import pyaudio
import numpy as np
from scipy import signal

# Constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 96000
TARGET_FREQ = 6000  # Your signal frequency
BANDWIDTH = 350    # Width of the frequency band to monitor

def get_audio_level(audio_data):
    """Calculate audio level from array of audio samples, focusing on 6kHz band"""
    if len(audio_data) == 0:
        return 0
        
    # Convert to float for calculations
    audio_float = audio_data.astype(np.float64)
    
    # Apply bandpass filter around 6kHz
    nyquist = RATE * 0.5
    low = (TARGET_FREQ - BANDWIDTH/2) / nyquist
    high = (TARGET_FREQ + BANDWIDTH/2) / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, audio_float)
    
    # Get absolute maximum value
    peak = np.abs(filtered).max()
    
    # Adjust threshold for noise floor
    if peak < 1e-3:  # Increased threshold to reduce noise
        return 0
        
    try:
        # Calculate RMS value of filtered signal
        rms = np.sqrt(np.mean(np.square(filtered)))
        # More aggressive scaling for underwater acoustics
        normalized = min(100, max(0, int(20 * np.log10(rms + 1e-10))))
        return normalized
    except (ValueError, RuntimeWarning):
        return 0

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Recording... (Press CTRL+C to stop)")

try:
    # Calculate how many chunks make up one second
    chunks_per_second = int(RATE / CHUNK)
    
    while True:
        # Collect one second of audio data
        levels = []
        for _ in range(chunks_per_second):
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            levels.append(get_audio_level(audio_data))
        
        # Average the levels over the second
        avg_level = int(np.mean(levels))
        print(f"Audio level: {avg_level}", end='\r')  # \r makes it update in place

except KeyboardInterrupt:
    print("\nStopped recording.")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()