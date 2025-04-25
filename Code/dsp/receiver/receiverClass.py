from typing import Dict, Tuple

import commpy.channelcoding.convcode as cc
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from config import (
    APPLY_AVERAGING_PREAMBLE,
    APPLY_BAKER_PREAMBLE,
    BINARY_BARKER,
    CONVOLUTIONAL_CODING,
    HAMMING_CODING,
    PATH_TO_WAV_FILE,
    
    BIT_RATE,
    CARRIER_FREQ,
    SAMPLE_RATE,
    PLOT_PREAMBLE_CORRELATION,
)

from encoding.hamming_codes import hamming_decode
from encoding.conv_encoding_scikit import conv_decode
from errors import PreambleNotFoundError
from scipy.io import wavfile
from visuals.visualization import create_processing_visualization

plt.style.use("ggplot")


def plot_wav_signal(sample_rate, wav_signal):
    time = np.linspace(0, len(wav_signal) / sample_rate, num=len(wav_signal))
    plt.figure(figsize=(10, 4))
    plt.plot(time, wav_signal, label="WAV Signal")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Waveform of WAV File")
    plt.legend()
    plt.grid()
    plt.show()
    print("Plotted WAV signal successfully.")


class Receiver:
    def __init__(self, band_pass: bool):
        _, self.wav_signal = wavfile.read(PATH_TO_WAV_FILE)
        self.bit_rate = BIT_RATE
        self.carrier_freq = CARRIER_FREQ
        self.band_pass = band_pass
        self.cutoff_freq = self.bit_rate * 5
        self.samples_per_symbol = int(SAMPLE_RATE / self.bit_rate)

    def _demodulate(self) -> Tuple[np.ndarray, Dict]:
        raise NotImplementedError("Subclasses must implement _demodulate")

    def bandpass_filter(self, input_signal: np.ndarray) -> np.ndarray:
        """Apply a bandpass filter around the carrier frequency"""
        nyquist = SAMPLE_RATE * 0.5
        order = 4
        low = (self.carrier_freq - self.bit_rate) / nyquist
        high = (self.carrier_freq + self.bit_rate) / nyquist

        b, a = signal.butter(order, [low, high], btype="band", analog=False)
        return signal.filtfilt(b, a, input_signal)

    def filter_signal(self, input_signal: np.ndarray) -> np.ndarray:
        nyquist = SAMPLE_RATE * 0.5
        order = 4
        cutoff = self.cutoff_freq / nyquist
        b, a = signal.butter(order, cutoff, btype="low", analog=False)
        return signal.filtfilt(b, a, input_signal)

    def remove_outliers(self, wave: np.ndarray) -> np.ndarray:
        wave = wave.copy()
        mu = np.mean(wave)
        sigma = np.std(wave)
        for i in range(len(wave)):
            if wave[i] > mu + 2 * sigma or wave[i] < mu - 2 * sigma:
                wave[i] = mu
        return wave

    def normalize_signal(self, signal_centered: np.ndarray) -> np.ndarray:
        return (signal_centered - np.min(signal_centered)) / (
            np.max(signal_centered) - np.min(signal_centered)
        )

    def threshold_signal(self, normalized_signal: np.ndarray) -> np.ndarray:
        # this is called hyperestesis thresholding, essentially you have a memory while checking
        low = 0.4
        high = 0.6
        thresholded = np.zeros_like(normalized_signal)
        state = 0
        for i in range(len(normalized_signal)):
            if state == 0 and normalized_signal[i] > low:
                state = 1
            elif state == 1 and normalized_signal[i] < high:
                state = 0
            thresholded[i] = state
        return thresholded

    def get_bits(self, thresholded_signal: np.ndarray) -> list:
        bits = []

        for i in range(
            0, len(thresholded_signal), self.samples_per_symbol
        ):
            mu = np.mean(
                thresholded_signal[i : i + self.samples_per_symbol]
            )
            bits.append(1 if mu > 0.5 else 0)
        return bits


    def remove_preamble_baker_code(self, bits, std_factor=4):
        correlation = signal.correlate(bits, BINARY_BARKER, mode="valid")
        threshold = np.mean(correlation) + std_factor * np.std(correlation)
        peak_indices, _ = signal.find_peaks(correlation, height=threshold, distance=20)
        
        if len(peak_indices) < 2:
            if std_factor > 1:
                return self.remove_preamble_baker_code(bits, std_factor - 0.1)
            else:
                return -1

        diff_in_peaks = np.diff(peak_indices)

        data_bits = []
        for i in range(len(peak_indices) - 1):
            # if abs(diff_in_peaks[i] - len_of_data_bits) <= 0:
            data_bits.append(bits[peak_indices[i] + len(BINARY_BARKER) : peak_indices[i + 1]])


        # NOTE: this is to plot the decodins of each entry of data bits
        print("Diff in peaks: ", diff_in_peaks)
        for i in range(len(data_bits)):
            if CONVOLUTIONAL_CODING:
                bits_array = np.array(data_bits[i])
                print(self.decode_bytes_to_bits(conv_decode(bits_array, None)[:-2]))
            elif HAMMING_CODING:
                print(self.decode_bytes_to_bits(hamming_decode(data_bits[i])))
            else:
                decoded_bits = self.decode_bytes_to_bits(data_bits[i])
                print(decoded_bits)
        if PLOT_PREAMBLE_CORRELATION:
            # NOTE: this plots the correlation of the preamble and the received signal
            plt.figure(figsize=(14, 8))
            plt.plot(
                correlation, color="#FF3300", label="Correlation Value", linewidth=2
            )
            plt.scatter(
                peak_indices,
                correlation[peak_indices],
                color="#000000",
                label="Detected Preambles",
                zorder=3,
                s=100,
                marker="D",
            )
            plt.axhline(
                threshold, color="gray", linestyle="--", label="Threshold", linewidth=2
            )
            plt.xlabel("Bits from received signal")
            plt.ylabel("Correlation Value")
            plt.title("Cross-Correlation with Preamble")
            plt.legend()
            plt.grid()
            plt.show()

        avg = [int(round((sum(col)) / len(col))) for col in zip(*data_bits)]
        if avg == []:
            avg = -1
        return avg


    def decode_bytes_to_bits(self, bits: list) -> str:
        if len(bits) % 8 != 0:
            remainder = len(bits) % 8
            bits += [0] * (8 - remainder)
        message = ""
        for i in range(0, len(bits), 8):
            byte = bits[i : i + 8]
            char = chr(int("".join(map(str, byte)), 2))
            if 32 <= ord(char) <= 126:
                message += char
            # else: # NOTE: all invalid characters will instead be "-", instead of just whitespace
            #     message += "-"
        return message

    def plot_simulation_steps(self):
        if self.wav_signal is None:
            print("No signal to visualize")
            return

        try:
            message, debug_info = self.decode()
        except Exception as e:
            print(f"could not plot {e}")
            return

        # Use the new visualization function
        fig = create_processing_visualization(self, message, debug_info)
        plt.show()
        return fig

    def set_len_of_data_bits(self, value):
        global len_of_data_bits
        len_of_data_bits = value


class NonCoherentReceiver(Receiver):
    def _demodulate(self) -> Tuple[np.ndarray, Dict]:
        if self.band_pass:
            self.wav_signal = self.bandpass_filter(self.wav_signal)

        fourier_transform_of_wav = np.fft.fft(self.wav_signal)
        length = len(fourier_transform_of_wav)
        fourier_analytic = np.zeros(length, dtype=complex)
        fourier_analytic[0] = fourier_transform_of_wav[0]
        fourier_analytic[1 : length // 2] = (
            2 * fourier_transform_of_wav[1 : length // 2]
        )
        fourier_analytic[length // 2 :] = 0
        analytic = np.fft.ifft(fourier_analytic)

        envelope = np.abs(analytic)
        filtered = self.filter_signal(envelope)
        return filtered, {
            "analytic": analytic,
            "envelope": envelope,
            "filtered": filtered,
        }

    def decode(self) -> Tuple[str, Dict]:
        filtered_signal, demod_debug = self._demodulate()
        cleaned_signal = self.remove_outliers(filtered_signal)
        normalized = self.normalize_signal(cleaned_signal)
        thresholded = self.threshold_signal(normalized)
        bits = self.get_bits(thresholded)

        # print("Received bits:", bits)

        if APPLY_AVERAGING_PREAMBLE:
            bits_without_preamble = self.remove_preamble_average(bits)
        elif APPLY_BAKER_PREAMBLE:
            bits_without_preamble = self.remove_preamble_baker_code(bits)
        else:
            bits_without_preamble = self.remove_preamble_naive(bits)

        if bits_without_preamble == -1:
            print("No preamble found, error was raised in receiverClass")
            raise PreambleNotFoundError("No preamble found in signal")

        if CONVOLUTIONAL_CODING:
            bits_without_preamble = conv_decode(bits_without_preamble)

        if HAMMING_CODING:
            bits_without_preamble = hamming_decode(bits_without_preamble)

        message = self.decode_bytes_to_bits(bits_without_preamble)
        debug_info = {
            **demod_debug,
            "normalized": normalized,
            "thresholded": thresholded,
            "bits_without_preamble": bits_without_preamble,
        }
        return message, debug_info


class CoherentReceiver(Receiver):
    def shift_signal(self, wav_signal: np.ndarray) -> np.ndarray:
        duration = len(wav_signal) / SAMPLE_RATE
        time_array = np.linspace(0, duration, len(wav_signal))
        coef = np.exp(-1j * 2 * np.pi * self.carrier_freq * time_array)
        return wav_signal * coef

    def _demodulate(self) -> Tuple[np.ndarray, Dict]:
        shifted = self.shift_signal(self.wav_signal)
        shifted_imag = -np.imag(shifted)
        filtered = self.filter_signal(shifted_imag)
        return filtered, {
            "shifted": shifted,
            "shifted_imag": shifted_imag,
            "filtered": filtered,
        }

    def decode(self) -> Tuple[str, Dict]:
        filtered_signal, demod_debug = self._demodulate()
        cleaned_signal = self.remove_outliers(filtered_signal)
        normalized = self.normalize_signal(cleaned_signal)
        thresholded = self.threshold_signal(normalized)
        bits = self.get_bits(thresholded)
        bits_without_preamble = self.remove_preamble_naive(bits)

        if CONVOLUTIONAL_CODING:
            bits = conv_decode(bits)

        message = self.decode_bytes_to_bits(bits_without_preamble)
        debug_info = {
            **demod_debug,
            "normalized": normalized,
            "thresholded": thresholded,
        }
        return message, debug_info


# Example usage
if __name__ == "__main__":
    # Hilbert method
    hilbert_receiver = NonCoherentReceiver.from_wav_file(PATH_TO_WAV_FILE)
    message, debug = hilbert_receiver.decode()
    print("Hilbert Decoded:", message)
    hilbert_receiver.plot_simulation_steps()

    # Shift-Imaginary method
    shift_receiver = CoherentReceiver.from_wav_file(PATH_TO_WAV_FILE)
    message, debug = shift_receiver.decode()
    print("Shift Decoded:", message)
    shift_receiver.plot_simulation_steps()
