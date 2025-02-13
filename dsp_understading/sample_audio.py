import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read

file_name = "jackhammer.wav"

Fs, data = read(file_name)
print(f"Sampling rate: {Fs} Hz")
print(f"Data type: {data.dtype}")
print(f"Data shape: {data.shape}")
print(f"Data: {data[:10]}")

#### Data is stereo so it has two channels
data_left_ch = data[:, 0]
data_right_ch = data[:, 1]

#### Create time array and plot the data
Ts = 1 / Fs
N = len(data_left_ch)
time = np.arange(0, N) * Ts

# plt.figure(figsize=(10, 6))
#
# plt.subplot(2, 1, 1)
# plt.plot(time, data_left_ch, "b")
# plt.title("Left channel")
#
# plt.subplot(2, 1, 2)
# plt.plot(time, data_right_ch, "r")
# plt.title("Right channel")
#
# plt.tight_layout()
# plt.show()

#### Plot the data in the frequency domain
frequencies_left = np.fft.fft(data_left_ch)
frequencies_right = np.fft.fft(data_right_ch)

N = len(frequencies_left)
### Center the frequencies
frequencies_left = np.fft.fftshift(frequencies_left)
frequencies_left = frequencies_left[: N // 2]

frequencies_right = np.fft.fftshift(frequencies_right)
frequencies_right = frequencies_right[: N // 2]

### plot the data
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(np.abs(frequencies_left), "b")
plt.title("Left channel")

plt.subplot(2, 1, 2)
plt.plot(np.abs(frequencies_right), "r")
plt.title("Right channel")

plt.tight_layout()
plt.show()
