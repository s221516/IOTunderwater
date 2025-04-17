import numpy as np
import matplotlib.pyplot as plt

# Define Barker Codes
barker_13 = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1])
barker_11 = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0])
barker_7 = np.array([1, 1, 1, 0, 0, 1, 0])

# Compute autocorrelations
autocorr_13 = np.correlate(barker_13, barker_13, mode='full')
autocorr_11 = np.correlate(barker_11, barker_11, mode='full')
autocorr_7 = np.correlate(barker_7, barker_7, mode='full')

# Adjust x-axis to center zero lag
lags_13 = np.arange(-len(barker_13) + 1, len(barker_13))
lags_11 = np.arange(-len(barker_11) + 1, len(barker_11))
lags_7 = np.arange(-len(barker_7) + 1, len(barker_7))

# Compute sidelobe values (excluding the main peak at lag=0)
sidelobes_13 = np.delete(autocorr_13, len(barker_13) - 1)
sidelobes_11 = np.delete(autocorr_11, len(barker_11) - 1)
sidelobes_7 = np.delete(autocorr_7, len(barker_7) - 1)
print("Sidelobe values (Barker-13):", sidelobes_13)
print("Sidelobe values (Barker-11):", sidelobes_11)
print("Sidelobe values (Barker-7):", sidelobes_7)

# Compute Peak-to-Sidelobe Ratio (PSLR)
pslr_13 = 20 * np.log10(np.max(autocorr_13) / np.max(np.abs(sidelobes_13)))
pslr_11 = 20 * np.log10(np.max(autocorr_11) / np.max(np.abs(sidelobes_11)))
pslr_7 = 20 * np.log10(np.max(autocorr_7) / np.max(np.abs(sidelobes_7)))
print("PSLR (Barker-13):", pslr_13, "dB")
print("PSLR (Barker-11):", pslr_11, "dB")
print("PSLR (Barker-7):", pslr_7, "dB")

# Align peaks by centering at zero lag
plt.figure(figsize=(12,8))
plt.plot(lags_13, autocorr_13, marker='o', linestyle='-', color='brown', linewidth=1.5, label="Barker-13")
plt.plot(lags_11, autocorr_11, marker='s', linestyle='-', color='green', linewidth=1.5, label="Barker-11")
plt.plot(lags_7, autocorr_7, marker='o', linestyle='-', color='blue', linewidth=1.5, label="Barker-7")

# Add horizontal lines for peak values
plt.axhline(y=np.max(autocorr_13), color='gray', linestyle='--', alpha=0.75)
plt.axhline(y=np.max(autocorr_11), color='gray', linestyle='--', alpha=0.75)
plt.axhline(y=np.max(autocorr_7), color='gray', linestyle='--', alpha=0.75)


plt.xlabel("Lag")
plt.ylabel("Auto-correlation Value")
plt.title("Auto-correlation function of Barker Codes")
plt.grid(True)
plt.xticks(np.arange(min(lags_13), max(lags_13) + 1, 1))
plt.yticks(np.arange(0, max(autocorr_13) + 1, 1))
plt.legend()
plt.show()