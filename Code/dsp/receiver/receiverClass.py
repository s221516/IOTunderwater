from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

from encoding.convolutional_encoding import *
from visuals.visualization import create_processing_visualization

from config_values import (
    CARRIER_FREQ,
    CUT_OFF_FREQ,
    PATH_TO_WAV_FILE,
    SAMPLE_RATE,
    SAMPLES_PER_SYMBOL,
    CONVOLUTIONAL_CODING,
    PREAMBLE_PATTERN,
    PREAMBLE_BASE
)
from scipy.io import wavfile

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
    def __init__(self, wav_signal: np.ndarray):
        self.wav_signal = wav_signal

    @classmethod
    def from_wav_file(cls, path: str):
        sample_rate, wav_signal = wavfile.read(path)
        return cls(wav_signal)
    
    def _demodulate(self) -> Tuple[np.ndarray, Dict]:
        raise NotImplementedError("Subclasses must implement _demodulate")

    def filter_signal(self, input_signal: np.ndarray) -> np.ndarray:
        nyquist = SAMPLE_RATE * 0.5
        order = 4
        cutoff = CUT_OFF_FREQ / nyquist
        b, a = signal.butter(order, cutoff, btype="low", analog=False)
        return signal.filtfilt(b, a, input_signal)

    @staticmethod
    def shift_signal(wav_signal: np.ndarray) -> np.ndarray:
        duration = len(wav_signal) / SAMPLE_RATE
        time_array = np.linspace(0, duration, len(wav_signal))
        coef = np.exp(-1j * 2 * np.pi * CARRIER_FREQ * time_array)
        return wav_signal * coef

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
        return self.hyperestesis_thresholding(normalized_signal, 0.5, 0.5)

    def hyperestesis_thresholding(
        self, normalized_signal: np.ndarray, low: float, high: float
    ) -> np.ndarray:
        thresholded = np.zeros_like(normalized_signal)
        state = 0
        for i in range(len(normalized_signal)):
            if state == 0 and normalized_signal[i] > high:
                state = 1
            elif state == 1 and normalized_signal[i] < low:
                state = 0
            thresholded[i] = state
        return thresholded

    def get_bits(self, thresholded_signal: np.ndarray) -> list:
        bits = []       
        for i in range(0, len(thresholded_signal), SAMPLES_PER_SYMBOL):
            mu = np.mean(thresholded_signal[i : i + SAMPLES_PER_SYMBOL])
            bits.append(1 if mu > 0.5 else 0)
        return bits

    def remove_preamble(self, bits):
        start_index = None
        end_index = None

        base_len = len(PREAMBLE_BASE)

        for i in range(0, len(bits) - (3 * base_len), 1):
            avg_pattern = []

            for j in range(base_len):
                sum_bit = bits[i + j] + bits[i + base_len + j] + bits[i + 2 * base_len + j]
                avg_bit = (sum_bit / 3)
                avg_pattern.append(1 if avg_bit > 0.5 else 0)
            
            if avg_pattern == PREAMBLE_BASE:
                start_index = i + 3 * base_len
                break
                
        if (start_index == None):
            return -1
        
        for i in range(start_index, len(bits) - (3 * base_len), 1):
            avg_pattern = []
            for j in range(base_len):
                sum_bit = bits[i + j] + bits[i + base_len + j] + bits[i + 2 * base_len + j]
                avg_bit = (sum_bit / 3)
                avg_pattern.append(1 if avg_bit > 0.5 else 0)
            
            if avg_pattern == PREAMBLE_BASE:
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
            message += chr(int("".join(map(str, byte)), 2))
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
        bits_without_preamble = self.remove_preamble(bits)
        if (bits_without_preamble == -1):
            print("No preamble found RecieverClass, decode method")
            raise TypeError

        if (CONVOLUTIONAL_CODING):
            bits_without_preamble = conv_decode(bits_without_preamble)

        message = self.decode_bytes_to_bits(bits_without_preamble)
        debug_info = {
            **demod_debug,
            "normalized": normalized,
            "thresholded": thresholded,
        }
        return message, debug_info


class CoherentReceiver(Receiver):
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
        bits_without_preamble = self.remove_preamble(bits)

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
