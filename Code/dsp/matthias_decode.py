import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from Code.dsp.transmitter_old import encode_and_modulate
from receiver import read_and_convert_wav_file
from config_values import (
    SAMPLE_RATE,
    CARRIER_FREQ,
    CUT_OFF_FREQ,
    SAMPLES_PER_BIT,
    MESSAGE,
)


def demodule_signal(signal_from_wav_file):

    # define time array
    time_arr = np.arange(len(signal_from_wav_file)) / SAMPLE_RATE
    # define coefficient to shift the signal to DC 0.
    coef = np.exp(-1j * 2 * np.pi * CARRIER_FREQ * time_arr)

    demod = signal_from_wav_file * coef
    return demod

def filter_signal(demod):
    """see reference look at answer : https://dsp.stackexchange.com/questions/49460/apply-low-pass-butterworth-filter-in-python"""
    nyquist_freq = SAMPLE_RATE * 0.5
    order = 4
    cutoff = CUT_OFF_FREQ / nyquist_freq
    # Not quite sure how these coefficients are calculated but has to be used to filter the signal
    coef_b, coef_a = signal.butter(order, cutoff, btype="low", analog=False)
    filtered_demodulated_signal = signal.filtfilt(coef_b, coef_a, demod)
    return filtered_demodulated_signal


def center_signal(filtered_demodulated_signal):
    # center signal
    centered_signal = filtered_demodulated_signal - np.mean(filtered_demodulated_signal)
    return centered_signal


def normalize_signal(centered_signal):
    """Norimalize from [-1, 1] to [0, 1]"""
    normalized_signal = (centered_signal - np.min(centered_signal)) / (
        np.max(centered_signal) - np.min(centered_signal)
    )
    return normalized_signal


def threshold_signal(normalized_signal):
    thresholded_signal = np.where(normalized_signal > 0, 1, 0)
    return thresholded_signal


def decode_message(thresholded_signal):

    def find_start_index(thresholded_signal):
        for i in range(len(thresholded_signal)):
            if thresholded_signal[i] != thresholded_signal[i - 1]:
                return i + SAMPLES_PER_BIT // 2
        return -1

    def get_bits_from_thresholded_signal(thresholded_signal):

        bits = []
        start_index = find_start_index(thresholded_signal)
        if start_index == -1:
            return print("No start index found")

        max_index = len(thresholded_signal) - SAMPLES_PER_BIT

        for i in range(start_index, max_index, SAMPLES_PER_BIT):
            bit_sample = thresholded_signal[i : i + SAMPLES_PER_BIT]
            bit = np.mean(bit_sample)
            bit_value = 1 if bit > 0 else 0
            bits.append(bit_value)

        return bits

    def get_message_from_bits(bits):

        message = ""
        for i in range(0, len(bits) - 7, 8):
            char_bits = bits[i : i + 8]
            if len(char_bits) == 8:
                char_code = int("".join(map(str, char_bits)), 2)
                if 32 <= char_code <= 126:  # Printable ASCII
                    message += chr(char_code)

        return message

    bits = get_bits_from_thresholded_signal(thresholded_signal)
    message = get_message_from_bits(bits)

    return message


def return_decoded_message(signal_from_wav_file):
    demod                       = demodule_signal(signal_from_wav_file)
    filtered_demodulated_signal = filter_signal(demod)
    # envelope                    = np.abs(filtered_demodulated_signal)
    centered_signal             = center_signal(filtered_demodulated_signal)
    normalized_signal           = normalize_signal(centered_signal)
    thresholded_signal          = threshold_signal(normalized_signal)

    # apply logic to get bits from thresholded_signal

    decoded_message = decode_message(thresholded_signal)

    return decoded_message


def debug_values(signal_from_wav_file):
    demod = demodule_signal(signal_from_wav_file)
    filtered_demodulated_signal = filter_signal(demod)
    centered_signal = center_signal(filtered_demodulated_signal)
    normalized_signal = normalize_signal(centered_signal)
    thresholded_signal = threshold_signal(centered_signal)
    return (
        demod,
        filtered_demodulated_signal,
        centered_signal,
        normalized_signal,
        thresholded_signal,
    )


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


if __name__ == "__main__":

    # read signal from wav file
    am_modulated = encode_and_modulate(MESSAGE)

    # decode message
    decoded_message = return_decoded_message(am_modulated)

    # debug values
    (
        demod,
        filtered_demodulated_signal,
        centered_signal,
        normalized_signal,
        thresholded_signal,
    ) = debug_values(am_modulated)

    plt.figure(figsize=(12, 16))
    plt.subplot(4, 2, 1)
    plot_wave_in_frequency_domain(am_modulated)
    plt.title("Signal from wav file")

    plt.subplot(4, 2, 3)
    plot_wave_in_frequency_domain(demod)
    plt.title("Demodulated signal")

    plt.subplot(4, 2, 5)
    plot_wave_in_frequency_domain(filtered_demodulated_signal)
    plt.title("Filtered demodulated signal")

    plt.subplot(4, 2, 7)
    plot_wave_in_frequency_domain(centered_signal)
    plt.title("Centered signal")

    plt.subplot(4, 2, 2)
    plot_wave_in_time_domain(am_modulated)
    plt.title("Signal from wav file")

    plt.subplot(4, 2, 4)
    plot_wave_in_time_domain(demod)
    plt.title("Demodulated signal")

    plt.subplot(4, 2, 6)
    plot_wave_in_time_domain(filtered_demodulated_signal)
    plt.title("Filtered demodulated signal")

    plt.subplot(4, 2, 8)
    plot_wave_in_time_domain(centered_signal)
    plt.title("Centered signal")

    plt.tight_layout()
    plt.show()

    plot_wave_in_time_domain(normalized_signal)
    plt.title("Normalized signal")
    plt.show()

    print(f"Message to be encoded: {MESSAGE}")
    print(f"Decoded message: {decoded_message}")
