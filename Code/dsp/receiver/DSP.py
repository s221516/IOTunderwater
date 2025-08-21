from cProfile import label
from logging import config
from typing import Dict, Tuple
import os
# import librosa
# import commpy.channelcoding.convcode as cc
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import config
# import config
from config import (
    BINARY_BARKER,
    BIPOLAR_BARKER,
    CONVOLUTIONAL_CODING,
    HAMMING_CODING,
)

from encoding.hamming_codes import hamming_decode
from encoding.conv_encoding_scikit import conv_decode
from scipy.io import wavfile


class Demodulator:
    def __init__(self, id: str, band_pass: bool):
        print(id)
        path = "Code/dsp/data/raw_data/" + id + ".wav"
        _, self.wav_signal = wavfile.read(path)
        self.bit_rate = config.BIT_RATE
        self.carrier_freq = config.CARRIER_FREQ
        self.band_pass = band_pass
        self.cutoff_freq = self.bit_rate 
        self.sample_rate = config.SAMPLE_RATE
        self.samples_per_symbol = int(self.sample_rate / self.bit_rate)
        self.filter_order = 4

    def _compute_len_of_bits(self, message):
        len_of_data_bits = len(message) * 8
        len_of_preamble = len(config.BINARY_BARKER)
        if config.CONVOLUTIONAL_CODING:
            len_of_data_bits = (len_of_data_bits * 2 + len_of_preamble + 4)
        elif config.HAMMING_CODING:
            len_of_data_bits = len_of_data_bits * 3 / 2 + len_of_preamble
        else:
            len_of_data_bits = len_of_data_bits + len_of_preamble

        print("Len of data bits (receiver): ", len_of_data_bits)
        return len_of_data_bits

    def _text_to_bits_list(text: str) -> list[int]:
        """Converts a string to a list of bits (8 bits per character, ASCII)."""
        bits_list = []
        for char_val in text:
            bits_list.extend([int(b) for b in bin(ord(char_val))[2:].zfill(8)])
        return bits_list

    def _find_first_preamble_start_in_bits(self, full_bitstream: list[int], std_factor_start=4, min_std_factor=1.0, step=0.1) -> int | None:
        """
        Internal helper to find the starting bit index of the first detected preamble.
        This performs its own correlation and peak finding.
        """
        full_bitstream = np.array(full_bitstream)
        BINARY_BARKER = np.array(config.BINARY_BARKER)
        print("Length of full_bitstream: ", len(full_bitstream))
        print("Length of BINARY_BARKER: ", len(BINARY_BARKER))
        
        if len(full_bitstream) < len(BINARY_BARKER):
            return None

        current_std_factor = std_factor_start
        while current_std_factor >= min_std_factor:
            correlation = signal.correlate(full_bitstream, BINARY_BARKER, mode="valid")
            print("Length of correlation: ", len(correlation))
            if len(correlation) == 0:
                return None # Should not happen if initial length check passed

            threshold = np.mean(correlation) + current_std_factor * np.std(correlation)
            # Find the first peak that meets the criteria
            peak_indices, _ = signal.find_peaks(correlation, height=threshold, distance=100)
            
            if peak_indices.size > 0:
                return peak_indices[0] # Return the index of the first peak found
            
            current_std_factor -= step
        
        return None # No preamble found after trying different std_factors

    def estimate_absolute_first_error_time(self, original_message_text: str, decoded_message_text: str, full_bitstream_from_get_bits: list[int] | None) -> float | None:
        """
        Estimates the absolute time of the first bit error from the start of the
        processed signal, by first finding the preamble in full_bitstream_from_get_bits.
        """
        if not original_message_text or not decoded_message_text or full_bitstream_from_get_bits is None:
            return None
        
        if self.samples_per_symbol == 0 or self.sample_rate == 0:
            return None

        first_preamble_start_index = self._find_first_preamble_start_in_bits(full_bitstream_from_get_bits)

        if first_preamble_start_index is None:
            # print("Debug: Preamble not found by estimate_absolute_first_error_time.")
            return None # Cannot establish payload start

        payload_start_offset_in_bitstream = first_preamble_start_index + len(BINARY_BARKER)

        original_bits = Demodulator._text_to_bits_list(original_message_text)
        print(f"Decoded message text: {decoded_message_text}")
        decoded_bits = Demodulator._text_to_bits_list(decoded_message_text)

        first_error_bit_index_in_payload = -1
        len_to_compare = min(len(original_bits), len(decoded_bits))

        for i in range(len_to_compare):
            if original_bits[i] != decoded_bits[i]:
                first_error_bit_index_in_payload = i
                break
        
        if first_error_bit_index_in_payload == -1 and len(original_bits) != len(decoded_bits):
            first_error_bit_index_in_payload = len_to_compare 

        if first_error_bit_index_in_payload != -1:
            absolute_error_bit_index = payload_start_offset_in_bitstream + first_error_bit_index_in_payload
            
            # Ensure the error index is within the bounds of the full bitstream
            if absolute_error_bit_index < len(full_bitstream_from_get_bits):
                error_time_seconds = (absolute_error_bit_index * self.samples_per_symbol) / self.sample_rate
                return error_time_seconds
            else:
                # print("Debug: Calculated absolute error bit index is out of bounds of the full bitstream.")
                return None
        
        return None # No error found in payload

    def _demodulate(self) -> Dict:
        raise NotImplementedError("Subclasses must implement _demodulate")

    def compute_average_power_of_signal(self) -> float:
        """Compute the average power of the signal"""
        return np.mean(self.wav_signal ** 2)

    def bandpass_filter(self, input_signal: np.ndarray) -> np.ndarray:
        """Apply a bandpass filter around the carrier frequency"""
        nyquist = self.sample_rate * 0.5
        order = 4
        low = (self.carrier_freq - self.bit_rate) / nyquist
        high = (self.carrier_freq + self.bit_rate) / nyquist

        b, a = signal.butter(order, [low, high], btype="band", analog=False)
        return signal.filtfilt(b, a, input_signal)

    def filter_signal(self, input_signal: np.ndarray) -> np.ndarray:
        nyquist = self.sample_rate * 0.5
        order = 8
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
        low = 0.5
        high = 0.5
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

    def remove_preamble_barker_code(self, bits, std_factor=4):
        
        #Bipolar mapping for Barker code
        bits = [1 if x == 1 else -1 for x in bits]

        correlation = signal.correlate(bits, BIPOLAR_BARKER, mode="valid") # old
        threshold = np.mean(correlation) + std_factor * np.std(correlation)
        peak_indices, _ = signal.find_peaks(correlation, height=threshold, distance=len_of_data_bits)
        if len(peak_indices) < 2:
            if std_factor > 1:
                return self.remove_preamble_barker_code(bits, std_factor - 0.1)
            else:
                print("Debug info: not finding preamble line 121")
                return [], [], []
            
        
        _, properties = signal.find_peaks(correlation, height=threshold, distance=len_of_data_bits)
        ### max height 
        max_peak = max(properties["peak_heights"])
        print("Max peak: ", max_peak)
        diff_in_peaks = np.diff(peak_indices)
        
        ### peak_indices with correlation = 9 
        print(type(properties["peak_heights"]))
        print("Peak heights: ", properties["peak_heights"])
        print("Peak indices: ", peak_indices)
        print(f"values at peak_indices: {correlation[peak_indices]}")
        
        data_bits_between_peaks = []
        data_bits_of_correct_len = []

        for i in range(len(peak_indices) - 1):
            data_section = bits[peak_indices[i] + len(BINARY_BARKER) : peak_indices[i + 1]]
            # Convert back to {0,1} format for decoding
            data_section_binary = [(1 if x == 1 else 0) for x in data_section]
            data_bits_between_peaks.append(bits[peak_indices[i] + len(BINARY_BARKER) : peak_indices[i + 1]])
            data_bits_of_correct_len.append(data_section_binary)

        # NOTE: this is to plot the decodins of each entry of data bits
        print("Diff in peaks: ", diff_in_peaks)
        for i in range(len(data_bits_of_correct_len)):
            if CONVOLUTIONAL_CODING:
                bits_array = np.array(data_bits_of_correct_len[i])
                print(self.decode_bytes_to_bits(conv_decode(bits_array, None)[:-2]))
            elif HAMMING_CODING:
                print(self.decode_bytes_to_bits(hamming_decode(data_bits_of_correct_len[i])))
            else:
                data_bits_of_correct_len[i] 
                decoded_bits = self.decode_bytes_to_bits(data_bits_of_correct_len[i])
                print(decoded_bits)
    
        avg = [int(round((sum(col)) / len(col))) for col in zip(*data_bits_of_correct_len)]

        return avg, data_bits_between_peaks, peak_indices

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

    
    
    def set_len_of_data_bits(self, value):
        global len_of_data_bits
        if isinstance(value, str):
            # If value is a string, compute length from message
            len_of_data_bits = self._compute_len_of_bits(value)
        else:
            # Otherwise, just set the value directly
            len_of_data_bits = value

class NonCoherentDemodulator(Demodulator):
    def _demodulate(self) -> Dict:
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
        return {
            "analytic": analytic,
            "envelope": envelope,
            "filtered": filtered
        }

    def demodulate(self) -> Tuple[str, Dict]:
        #Pure DSP
        demod_debug = self._demodulate()
        cleaned_signal = self.remove_outliers(demod_debug["filtered"])
        normalized = self.normalize_signal(cleaned_signal)
        thresholded = self.threshold_signal(normalized)
        bits = self.get_bits(thresholded)

        ## Barker code detection & extraction of message in bits
        estimated_message_in_bits, data_bits_between_peaks, peak_indices = self.remove_preamble_barker_code(bits)
        
        if estimated_message_in_bits == []:
            return "Error: Preamble not found", {}

        #Removal of decoding
        if CONVOLUTIONAL_CODING:
            estimated_message_in_bits = conv_decode(estimated_message_in_bits)

        if HAMMING_CODING:
            estimated_message_in_bits = hamming_decode(estimated_message_in_bits)

        message = self.decode_bytes_to_bits(estimated_message_in_bits)
        debug_info = {
            **demod_debug,
            "data_bits_between_peaks": data_bits_between_peaks,
            "normalized": normalized,
            "thresholded": thresholded,
            "estimated_message_in_bits": estimated_message_in_bits,
            "peak_indices": peak_indices
        }


        return message, debug_info

class CoherentDemodulator(Demodulator):
    def demodulate(self) -> Tuple[str, Dict]:
        #To be implemented
        return "Error: Coherent demodulation not implemented", {}

