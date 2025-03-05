from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from config_values import (
    CARRIER_FREQ,
    CUT_OFF_FREQ,
    PATH_TO_WAV_FILE,
    SAMPLE_RATE,
    SAMPLES_PER_SYMBOL,
)
from scipy.io import wavfile


def get_wav_signal(path: str):
    _, wav_signal = wavfile.read(path)
    return wav_signal


def hyperestesis_thresholding(normalized_signal, low=0.4, high=0.6):
    thresholded = np.zeros_like(normalized_signal)
    state = 0  # 0=low, 1=high
    for i in range(len(normalized_signal)):
        if state == 0 and normalized_signal[i] > high:
            state = 1
        elif state == 1 and normalized_signal[i] < low:
            state = 0
        thresholded[i] = state
    return thresholded


def shift_signal(wav_signal):
    duration_of_wav_signal = len(wav_signal) / SAMPLE_RATE
    time_array = np.linspace(0, duration_of_wav_signal, len(wav_signal))
    coef = np.exp(-1j * 2 * np.pi * CARRIER_FREQ * time_array)

    return wav_signal * coef


def filter_signal(wav_shifted_signal):
    # TODO LOOK UP PYSDR FILTERS CHAPTER
    """see reference look at answer : https://dsp.stackexchange.com/questions/49460/apply-low-pass-butterworth-filter-in-python"""
    nyquist_freq = SAMPLE_RATE * 0.5
    order = 4
    cutoff = CUT_OFF_FREQ / nyquist_freq
    # Not quite sure how these coefficients are calculated but has to be used to filter the signal
    coef_b, coef_a = signal.butter(order, cutoff, btype="low", analog=False)
    filtered_demodulated_signal = signal.filtfilt(coef_b, coef_a, wav_shifted_signal)
    return filtered_demodulated_signal


def normalize_signal(signal_centered):
    # Scale to [0, 1]
    normalized = (signal_centered - np.min(signal_centered)) / (
        np.max(signal_centered) - np.min(signal_centered)
    )
    return normalized


def threshold_signal(normalized_signal):
    low = 0.5
    high = 0.5
    # TODO: possible to fine tune the threshold values for small improvements --> dont go below 0.4 or above 0.6
    return hyperestesis_thresholding(normalized_signal, low, high)


def get_bits(thresholded_signal):
    bits = []
    for i in range(0, len(thresholded_signal), SAMPLES_PER_SYMBOL):
        mu = np.mean(thresholded_signal[i : i + SAMPLES_PER_SYMBOL])
        bits.append(1 if mu > 0.5 else 0)
    return bits


def decode_bits(bits):
    if len(bits) % 8 != 0:
        print(f"len of bits {len(bits)}")
        print(f"len of bits % 8 = {len(bits) % 8}")
        print("ERROR: Number of bits is not a multiple of 8")
        print("Number of bits:", len(bits))
        print("Therefore cannot decode message")
        return ""

    decoded_message = ""
    for i in range(0, len(bits), 8):
        byte = bits[i : i + 8]
        byte_str = "".join(str(bit) for bit in byte)
        decoded_message += chr(int(byte_str, 2))

    return decoded_message


def decode_message(thresholded_signal):
    bits = get_bits(thresholded_signal)
    print("Bits:", bits)
    decoded_message = decode_bits(bits)
    return decoded_message

def remove_outliers(wave):
    mu = np.mean(wave)
    sigma = np.std(wave)
    
    for i in range(0, len(wave)):
        if wave[i] > mu + 3*sigma or wave[i] < mu - 3*sigma:
            wave[i] = mu
    return wave
def decode_wav_signal(wav_signal):
    # analytic_signal = signal.hilbert(wav_signal)
    # envelope = np.abs(analytic_signal)
    #
    shifted_signal = shift_signal(wav_signal)
    shifted_signal = -np.imag(shifted_signal)
    filtered_signal = filter_signal(shifted_signal)
    filtered_signal = remove_outliers(filtered_signal)

    # Normalize the demodulated signal
    normalized_signal = (filtered_signal - np.min(filtered_signal)) / (
        np.max(filtered_signal) - np.min(filtered_signal)
    )
    thresholded_signal = threshold_signal(normalized_signal)
    decoded_message = decode_message(thresholded_signal)

    # Update plots to show the imaginary part
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(wav_signal, label="Original")
    plt.title("Original Signal")
    plt.subplot(4, 1, 2)
    plt.plot(filtered_signal, label="Filtered (Imaginary)")
    plt.title("Demodulated Signal (Imaginary Part)")
    plt.subplot(4, 1, 3)
    plt.plot(normalized_signal, label="Normalized")
    plt.title("Normalized Signal")
    plt.subplot(4, 1, 4)
    plt.plot(thresholded_signal, label="Thresholded")
    plt.plot(normalized_signal, label="Normalized", alpha=0.5)
    plt.title("Thresholded Signal")
    plt.legend()
    plt.show()

    return decoded_message, {
        "filtered_signal": filtered_signal,
        "normalized_signal": normalized_signal,
        "thresholded_signal": thresholded_signal,
    }


def plot_wave_in_frequency_domain(wave):
    wave_f = np.fft.fft(wave)
    wave_f = np.fft.fftshift(wave_f)
    frequencies_x_axis = np.arange(
        SAMPLE_RATE / -2, SAMPLE_RATE / 2, SAMPLE_RATE / len(wave)
    )

    frequency_magnitudes = np.abs(wave_f)

    plt.plot(frequencies_x_axis, frequency_magnitudes, ".-", "b", alpha=0.5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Frequency Magnitude")
    plt.title("Frequency Domain")


def plot_wave_in_time_domain(wave):
    time_array = np.arange(len(wave)) / SAMPLE_RATE
    plt.plot(time_array, wave, "orange", alpha=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")


def visualization_to_debug(wav_signal: np.array, debug_values: Dict):
    plt.figure()
    plt.subplot(4, 2, 1)
    plot_wave_in_time_domain(wav_signal)
    plt.title("Original Signal")

    plt.subplot(4, 2, 2)
    plot_wave_in_frequency_domain(wav_signal)
    plt.title("Original Signal")

    plt.subplot(4, 2, 3)
    plot_wave_in_time_domain(debug_values["envelope"])
    plt.title("Envelope")

    plt.subplot(4, 2, 4)
    plot_wave_in_frequency_domain(debug_values["envelope"])
    plt.title("Envelope")
    plt.subplot(4, 2, 5)
    plot_wave_in_time_domain(debug_values["filtered_signal"])
    plt.title("Filtered Signal")

    plt.subplot(4, 2, 6)
    plot_wave_in_frequency_domain(debug_values["filtered_signal"])
    plt.title("Filtered Signal")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    wav_signal = get_wav_signal(PATH_TO_WAV_FILE)
    decoded_wav_signal, debug_values = decode_wav_signal(wav_signal)
    time_array = np.linspace(
        0, (len(decoded_wav_signal) / SAMPLE_RATE), len(decoded_wav_signal)
    )
    # print(
    #     f" following is the decoded message:{decoded_wav_signal}",
    #     len(decoded_wav_signal),
    # )
    # visualization_to_debug(wav_signal, debug_values)
