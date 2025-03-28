# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import correlate

# # Step 1: Define the Barker Code (7-bit)
# barker_7 = np.array([1, 1, 1, -1, -1, 1, -1])  # Original Barker sequence
# BINARY_BARKER = (barker_7 + 1) // 2  # Convert to binary (0s and 1s)
# print("Barker Code (Binary):", BINARY_BARKER)
# BINARY_BARKER = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]

# # Step 2: Simulate a transmitted signal (including noise & errors)
# transmitted_signal = np.concatenate([np.zeros(10), BINARY_BARKER, np.array([1,1,1,0,1,1,1,1])])  # Add padding
# received_signal = transmitted_signal.copy()
# received_signal[15] = 1 - received_signal[15]  # Introduce a bit error
# print("Received Signal:", received_signal)

# # Step 3: Perform cross-correlation for detection
# correlation = correlate(received_signal, BINARY_BARKER, mode='valid')
# peak_index = np.argmax(correlation)  # Find the best match
# print(peak_index)

# # Step 4: Detect the preamble
# threshold = 3  # Adjust based on noise tolerance
# if correlation[peak_index] >= threshold:
#     print(f"Preamble detected at index {peak_index}")
# else:
#     print("Preamble not detected")

# print(received_signal[peak_index + len(BINARY_BARKER):])

# # Step 5: Plot correlation result
# plt.figure(figsize=(10, 4))
# plt.subplot(2, 1, 1)
# plt.stem(received_signal, linefmt='b-', markerfmt='bo', basefmt='r-')
# plt.title("Received Signal with Barker Code")
# plt.xlabel("Sample Index")
# plt.ylabel("Amplitude")

# plt.subplot(2, 1, 2)
# plt.plot(correlation, label="Cross-correlation Output", color='g')
# plt.axvline(peak_index, color='r', linestyle='--', label="Detected Preamble")
# plt.legend()
# plt.title("Barker Code Detection via Correlation")
# plt.xlabel("Sample Index")
# plt.ylabel("Correlation Value")

# plt.tight_layout()
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Define Barker Codes
barker_13 = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1])
barker_7 = np.array([1, 1, 1, -1, -1, 1, -1])

# Compute autocorrelations
autocorr_13 = np.correlate(barker_13, barker_13, mode='full')
autocorr_7 = np.correlate(barker_7, barker_7, mode='full')

# Adjust x-axis to center zero lag
lags_13 = np.arange(-len(barker_13) + 1, len(barker_13))
lags_7 = np.arange(-len(barker_7) + 1, len(barker_7))

# Compute sidelobe values (excluding the main peak at lag=0)
sidelobes_13 = np.delete(autocorr_13, len(barker_13) - 1)
sidelobes_7 = np.delete(autocorr_7, len(barker_7) - 1)
print("Sidelobe values (Barker-13):", sidelobes_13)
print("Sidelobe values (Barker-7):", sidelobes_7)

# Compute Peak-to-Sidelobe Ratio (PSLR)
pslr_13 = 20 * np.log10(np.max(autocorr_13) / np.max(np.abs(sidelobes_13)))
pslr_7 = 20 * np.log10(np.max(autocorr_7) / np.max(np.abs(sidelobes_7)))
print("PSLR (Barker-13):", pslr_13, "dB")
print("PSLR (Barker-7):", pslr_7, "dB")


# Align peaks by centering at zero lag
plt.figure(figsize=(9,5))
plt.plot(lags_13, autocorr_13, marker='o', linestyle='-', color='brown', linewidth=1.5, label="Barker-13")
plt.plot(lags_7, autocorr_7, marker='o', linestyle='-', color='blue', linewidth=1.5, label="Barker-7")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation function of Barker Codes")
plt.grid(True)
plt.xticks(np.arange(min(lags_13), max(lags_13) + 1, 1))
plt.yticks(np.arange(0, 10, 1))
plt.legend()
plt.show()