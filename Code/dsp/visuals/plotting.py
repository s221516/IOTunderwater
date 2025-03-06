import matplotlib.pyplot as plt
import numpy as np
from config_values import SAMPLE_RATE


# Modified plotting functions to accept an axis parameter
def plot_wave_in_frequency_domain(wave, ax=None, color="b"):
    if ax is None:
        ax = plt.gca()

    wave_f = np.fft.fft(wave)
    wave_f = np.fft.fftshift(wave_f)
    frequencies_x_axis = np.arange(
        SAMPLE_RATE / -2, SAMPLE_RATE / 2, SAMPLE_RATE / len(wave)
    )

    frequency_magnitudes = np.abs(wave_f)

    ax.plot(frequencies_x_axis, frequency_magnitudes, ".-", color=color, alpha=0.5)


def plot_wave_in_time_domain(wave, l: str, ax=None, color="orange"):
    if ax is None:
        ax = plt.gca()

    time_array = np.arange(len(wave)) / SAMPLE_RATE
    ax.plot(time_array, wave, color=color, label=l, alpha=0.5)
    plt.grid(True)
