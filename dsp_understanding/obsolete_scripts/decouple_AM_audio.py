import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

# Load the modulated WAV file
sample_rate, modulated_wave = wav.read("hello_world_am.wav")
modulated_wave = modulated_wave.astype(np.float32) / 32767.0  # Normalize

# Define known carrier frequency
carrier_freq = 10000  # Hz
bit_rate = 16  # Bits per second


# Apply bandpass filter to isolate carrier
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, data)


filtered_signal = bandpass_filter(
    modulated_wave, carrier_freq - 2, carrier_freq + 2, sample_rate
)

# Demodulate using Hilbert Transform (Extract Envelope)
analytic_signal = np.real(signal.hilbert(filtered_signal))
envelope = abs(analytic_signal)


# Low-pass filter to smooth the envelope
def lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype="low")
    return signal.filtfilt(b, a, data)


demodulated_signal = lowpass_filter(envelope, bit_rate, sample_rate)

# Normalize demodulated signal
demodulated_signal = (demodulated_signal - np.min(demodulated_signal)) / (
    np.max(demodulated_signal) - np.min(demodulated_signal)
)

# Thresholding to extract binary data
threshold = 0.5  # Midpoint threshold between high and low states
binary_signal = (demodulated_signal > threshold).astype(int)

# Extract bits at the correct timing (bit sampling)
samples_per_bit = int(sample_rate / bit_rate)
binary_data = binary_signal[::samples_per_bit]  # Sample at bit rate intervals

# Convert binary to ASCII text
binary_string = "".join(
    map(str, binary_data[: len(binary_data) // 8 * 8])
)  # Ensure full bytes
ascii_chars = [
    chr(int(binary_string[i : i + 8], 2)) for i in range(0, len(binary_string), 8)
]
decoded_text = "".join(ascii_chars)

if len(decoded_text) == 0:
    print("No data extracted from the audio file")

# Print the extracted message
print(type(decoded_text))
print(f"length of the message is {len(decoded_text)}")
print(f"Decoded Message: {decoded_text}")

# Plot results
plt.figure(figsize=(10, 4))
plt.subplot(3, 1, 1)
plt.plot(modulated_wave[: 5 * sample_rate], label="Modulated Signal", alpha=0.7)
plt.title("Received Modulated Signal")

plt.subplot(3, 1, 2)
plt.plot(demodulated_signal[: 5 * sample_rate], "g", label="Demodulated Envelope")
plt.axhline(int(threshold), color="r", linestyle="--", label="Threshold")
plt.title("Demodulated Envelope")

plt.subplot(3, 1, 3)
plt.step(range(len(binary_data)), binary_data, "k", label="Extracted Binary Data")
plt.title("Recovered Binary Data")

plt.tight_layout()
plt.show()
