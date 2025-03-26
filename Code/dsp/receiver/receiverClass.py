from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
from errors import PreambleNotFoundError

from encoding.convolutional_encoding import *
from encoding.hamming_codes import hamming_decode
from visuals.visualization import create_processing_visualization
from config import (
    PATH_TO_WAV_FILE,
    SAMPLE_RATE,
    CONVOLUTIONAL_CODING,
    PREAMBLE_PATTERN,
    PREAMBLE_BASE, 
    APPLY_AVERAGING_PREAMBLE,
    REPETITIONS,
    BINARY_BARKER, 
    APPLY_BAKER_PREAMBLE, 
    LEN_OF_DATA_BITS,
    HAMMING_CODING, 
    BAND_PASS_FILTER
)


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
    
    def __init__(self, bit_rate: int, carrier_freq : int, band_pass: bool):
        
        _, self.wav_signal = wavfile.read(PATH_TO_WAV_FILE)
        self.bit_rate           = bit_rate
        self.carrier_freq       = carrier_freq
        self.band_pass          = band_pass
        self.cutoff_freq        = (carrier_freq + bit_rate) // 2
        self.samples_per_symbol = int(SAMPLE_RATE / bit_rate) 

    def _demodulate(self) -> Tuple[np.ndarray, Dict]:
        raise NotImplementedError("Subclasses must implement _demodulate")

    def bandpass_filter(self, input_signal: np.ndarray) -> np.ndarray:
        """Apply a bandpass filter around the carrier frequency"""
        nyquist = SAMPLE_RATE * 0.5
        order = 4
        # Define bandpass range around carrier frequency
        low = (self.carrier_freq - self.bit_rate) / nyquist
        high = (self.carrier_freq + self.bit_rate) / nyquist
        b, a = signal.butter(order, [low, high], btype='band', analog=False)
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
        for i in range(0, len(thresholded_signal), self.samples_per_symbol):
            mu = np.mean(thresholded_signal[i : i + self.samples_per_symbol])
            bits.append(1 if mu > 0.5 else 0)
        return bits

    def remove_preamble_average(self, bits):
        start_index = None
        end_index = None

        base_len = len(PREAMBLE_BASE)

        for i in range(0, len(bits) - (REPETITIONS * base_len), 1):
            avg_pattern = []

            for j in range(base_len):
                sum_bit = bits[i + j] + bits[i + base_len + j] + bits[i + 2 * base_len + j] + bits[i + 3 * base_len + j] + bits[i + 4 * base_len + j] 
                avg_bit = (sum_bit / REPETITIONS)
                avg_pattern.append(1 if avg_bit > 0.5 else 0)
            
            if avg_pattern == PREAMBLE_BASE:
                start_index = i + REPETITIONS * base_len
                break
                
        if (start_index == None):
            return -1
        
        for i in range(start_index, len(bits) - (REPETITIONS * base_len), 1):
            avg_pattern = []
            for j in range(base_len):
                sum_bit = bits[i + j] + bits[i + base_len + j] + bits[i + 2 * base_len + j] + bits[i + 3 * base_len + j] + bits[i + 4 * base_len + j]
                avg_bit = (sum_bit / REPETITIONS)
                avg_pattern.append(1 if avg_bit > 0.5 else 0)
            
            if avg_pattern == PREAMBLE_BASE:
                end_index = i
                break
            
            return bits[start_index:end_index]
    
    def remove_preamble_baker_code(self, bits, std_factor = 4):
        correlation = signal.correlate(bits, BINARY_BARKER, mode='valid')
        threshold = np.mean(correlation) + std_factor * np.std(correlation)
        # threshold = 9
        peak_indices, _ = signal.find_peaks(correlation, height=threshold, distance=LEN_OF_DATA_BITS)
        
        if len(peak_indices) < 2:
            if std_factor > 1 : 
                return self.remove_preamble_baker_code(bits, std_factor - 0.1)   
            else:
                return -1
        
        diff_in_peaks = np.diff(peak_indices)
        print("Diff in peaks: ", diff_in_peaks)

        data_bits = []
        for i in range(len(peak_indices) - 1):
            if LEN_OF_DATA_BITS == diff_in_peaks[i]: # NOTE: we tested less than equal but it makes a big difference with just a few bits, it needs to be exactly equal, as one bit makes a huge difference when decoding
                data_bits.append(bits[peak_indices[i] + len(BINARY_BARKER): peak_indices[i+1]])


        for i in range(len(data_bits)):
            if CONVOLUTIONAL_CODING:
                print(self.decode_bytes_to_bits(conv_decode(data_bits[i])))
            elif HAMMING_CODING:
                print(self.decode_bytes_to_bits(hamming_decode(data_bits[i])))
            else:
                bits = self.decode_bytes_to_bits(data_bits[i])
                print(bits, len(bits))

        avg = [int(round((sum(col))/len(col))) for col in zip(*data_bits)]

        # plt.figure(figsize=(10, 4))
        # plt.plot(correlation, label="Cross-Correlation")
        # plt.scatter(peak_indices, correlation[peak_indices], color='red', label="Detected Preambles", zorder=3)
        # plt.axhline(threshold, color='gray', linestyle='--', label="Threshold")
        # plt.xlabel("Index of bit")
        # plt.ylabel("Correlation Value")
        # plt.title("Cross-Correlation with Preamble")
        # plt.legend()
        # plt.grid()
        # plt.show()

        return avg

    def remove_preamble_naive(self, bits):        
        start_index = None
        end_index = None
        for i in range(0, len(bits), 1):
            if bits[i: i+len(PREAMBLE_PATTERN)] == PREAMBLE_PATTERN:
                start_index = i + len(PREAMBLE_PATTERN)
                break
                
        if (start_index == None):
            return -1
        
        for i in range(start_index, len(bits), 1):
            if bits[i: i+len(PREAMBLE_PATTERN)] == PREAMBLE_PATTERN:
                end_index = i
                break
        
        return bits[start_index:end_index]

    def decode_bytes_to_bits(self, bits: list) -> str:
        if len(bits) % 8 != 0:
            remainder = len(bits) % 8
            bits += ([0] * (8 - remainder))
        message = ""
        for i in range(0, len(bits), 8):
            byte = bits[i : i + 8]
            char = chr(int("".join(map(str, byte)), 2))
            if 32 <= ord(char) <= 126:
                message += char
            else: # NOTE: all invalid characters will instead be "-", instead of just whitespace
                message += "-"
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


class NonCoherentReceiver(Receiver):
    def _demodulate(self) -> Tuple[np.ndarray, Dict]:
        if (self.band_pass):
            bandpassed = self.bandpass_filter(self.wav_signal)
            analytic = signal.hilbert(bandpassed)
        else:   
            analytic = signal.hilbert(self.wav_signal)
            
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

        if (APPLY_AVERAGING_PREAMBLE):
            bits_without_preamble = self.remove_preamble_average(bits)
        elif (APPLY_BAKER_PREAMBLE):
            bits_without_preamble = self.remove_preamble_baker_code(bits)
        else:
            bits_without_preamble = self.remove_preamble_naive(bits)
        
        if (bits_without_preamble == -1):
            print("No preamble found, error was raised in receiverClass")       
            raise PreambleNotFoundError("No preamble found in signal")

        if (CONVOLUTIONAL_CODING):
            bits_without_preamble = conv_decode(bits_without_preamble)

        if (HAMMING_CODING):
            bits_without_preamble = hamming_decode(bits_without_preamble)

        message = self.decode_bytes_to_bits(bits_without_preamble)
        debug_info = {
            **demod_debug,
            "normalized": normalized,
            "thresholded": thresholded,
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

        if (CONVOLUTIONAL_CODING):
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
