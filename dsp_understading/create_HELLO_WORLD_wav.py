import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav

# Define parameters
sample_rate = 44100  # 44.1 kHz standard audio sampling rate
carrier_freq = 10000  # Hz
bit_rate = 1000  # Bits per second (slow for clarity)
duration_per_bit = 1 / bit_rate  # seconds per bit

# Convert "HELLO WORLD" to binary ASCII
text = "HELLO WORLD"
binary_message = "".join(format(ord(c), "08b") for c in text)  # 8-bit ASCII encoding
print(f"Binary Representation: {binary_message}")

# Generate time array
total_duration = len(binary_message) * duration_per_bit
t = np.linspace(0, total_duration, int(sample_rate * total_duration), endpoint=False)

# Create the square wave modulation signal
modulation_signal = np.repeat(
    [int(b) for b in binary_message], int(sample_rate * duration_per_bit)
)
modulation_signal = np.pad(
    modulation_signal, (0, len(t) - len(modulation_signal)), "constant"
)

# Normalize modulation signal to 0.3 - 1 (so carrier is not fully suppressed at 0s)
modulation_signal = (
    0.3 + 0.7 * modulation_signal
)  # Adjust amplitude between 0.3 (low) and 1 (high)

# Create the carrier wave (10 Hz sine wave)
carrier_wave = np.sin(2 * np.pi * carrier_freq * t)

# Apply amplitude modulation (AM)
modulated_wave = modulation_signal * carrier_wave

# Normalize to int16 range and save
modulated_wave = (modulated_wave * 32767).astype(
    np.int16
)  # Convert to 16-bit PCM format
wav.write("hello_world_am.wav", sample_rate, modulated_wave)

# Plot the modulation signal and modulated wave
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t[: int(2 * sample_rate)], modulation_signal[: int(2 * sample_rate)], "r")
plt.title("Square Wave Modulation Signal (First 2 Seconds)")
plt.subplot(2, 1, 2)
plt.plot(t[: int(2 * sample_rate)], modulated_wave[: int(2 * sample_rate)], "b")
plt.title("AM Modulated Signal (First 2 Seconds)")
plt.tight_layout()
plt.show()
