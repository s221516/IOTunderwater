import matplotlib.pyplot as plt
import numpy as np

# determine a frequency
Fs = 1000  # sampling frequency
N = 1000  # number of samples

# Create a time vector
t = np.linspace(0, 1, N)
# Create a signal
s = np.sin(2 * np.pi * Fs * t) + 0.5 * np.sin(0.15 * np.pi * 2 * Fs * t)
s = np.hamming(N) * s

# Compute the Fourier Transform shifted to have 0 frequency in the middle
S = np.fft.fftshift(np.fft.fft(s))
# Compute the real part and the imaginary part of the Fourier Transform
S_mag = np.abs(S)
S_phase = np.angle(S)

# Create a frequency vector
f = np.arange(-Fs / 2, Fs / 2, Fs / N)

# Plot the Fourier Transform in 2 subplots (magnitude and phase)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(f, S_mag)
plt.title("Magnitude")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(f, S_phase)
plt.title("Phase")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase")
plt.grid()

plt.tight_layout()
plt.show()
