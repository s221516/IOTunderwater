from logging import config
from typing import Dict, Tuple

import librosa
import commpy.channelcoding.convcode as cc
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import config
# import config
from config import (
    APPLY_BAKER_PREAMBLE,
    BINARY_BARKER,
    CONVOLUTIONAL_CODING,
    HAMMING_CODING,
    PATH_TO_WAV_FILE,
    PLOT_PREAMBLE_CORRELATION,
    IS_ID_SPECIFIED
)

from encoding.hamming_codes import hamming_decode
from encoding.conv_encoding_scikit import conv_decode
from errors import PreambleNotFoundError
from scipy.io import wavfile
from visuals.visualization import create_processing_visualization, create_frequency_domain_visualization

# plt.style.use("ggplot")


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
    def __init__(self, id: str, band_pass: bool):
        _, self.wav_signal = wavfile.read("Code/dsp/data/raw_data/" + id + ".wav")
        self.bit_rate = config.BIT_RATE
        self.carrier_freq = config.CARRIER_FREQ
        self.band_pass = band_pass
        self.cutoff_freq = self.bit_rate * 5
        self.sample_rate = config.SAMPLE_RATE
        self.samples_per_symbol = int(self.sample_rate / self.bit_rate)
        self.filter_order = 4

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

        original_bits = Receiver._text_to_bits_list(original_message_text)
        print(f"Decoded message text: {decoded_message_text}")
        decoded_bits = Receiver._text_to_bits_list(decoded_message_text)

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

    def _demodulate(self) -> Tuple[np.ndarray, Dict]:
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
        low = 0.5
        print("Low: ", low)
        high = 0.5
        print("High: ", high)
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


    def remove_preamble_baker_code(self, bits, std_factor=4):
        correlation = signal.correlate(bits, BINARY_BARKER, mode="valid") # old
        # correlation = signal.correlate(BINARY_BARKER, bits, mode="same") # new
        threshold = np.mean(correlation) + std_factor * np.std(correlation)
        peak_indices, _ = signal.find_peaks(correlation, height=threshold, distance=100)
        if len(peak_indices) < 2:
            if std_factor > 1:
                return self.remove_preamble_baker_code(bits, std_factor - 0.1)
            else:
                print("Debug info: not finding preamble line 121")
                return -1, [], []
            
        
        _, properties = signal.find_peaks(correlation, height=threshold, distance=100)
        ### max height 
        max_peak = max(properties["peak_heights"])
        print("Max peak: ", max_peak)
        diff_in_peaks = np.diff(peak_indices)
        
        ### peak_indices with correlation = 9 
        print(type(properties["peak_heights"]))
        print("Peak heights: ", properties["peak_heights"])
        print("Peak indices: ", peak_indices)
        print(f"values at peak_indices: {correlation[peak_indices]}")
        
        all_data_bits            = []
        data_bits_of_correct_len = []
        for i in range(len(peak_indices) - 1):
            all_data_bits.append(bits[peak_indices[i] + len(BINARY_BARKER) : peak_indices[i + 1]])
            # if abs(diff_in_peaks[i] - len_of_data_bits) == 0:
            data_bits_of_correct_len.append(bits[peak_indices[i] + len(BINARY_BARKER) : peak_indices[i + 1]])

        # NOTE: this is to plot the decodins of each entry of data bits
        print("Diff in peaks: ", diff_in_peaks)
        for i in range(len(data_bits_of_correct_len)):
            if CONVOLUTIONAL_CODING:
                bits_array = np.array(data_bits_of_correct_len[i])
                print(self.decode_bytes_to_bits(conv_decode(bits_array, None)[:-2]))
            elif HAMMING_CODING:
                print(self.decode_bytes_to_bits(hamming_decode(data_bits_of_correct_len[i])))
            else:
                decoded_bits = self.decode_bytes_to_bits(data_bits_of_correct_len[i])
                print(decoded_bits)
        if PLOT_PREAMBLE_CORRELATION:
            # NOTE: this plots the correlation of the preamble and the received signal
            plt.figure(figsize=(10, 6))
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
            plt.title("Cross-Correlation with Preamble With Bandpass (BER: 0.54)")
            plt.legend()
            plt.grid()
            plt.show()

        avg = [int(round((sum(col)) / len(col))) for col in zip(*data_bits_of_correct_len)]
        if avg == []:
            avg = -1
        return avg, all_data_bits, peak_indices


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

    def plot_signal(self):
        
        ### plot the original signal in time and frequency domain
        if self.wav_signal is None:
            print("No signal to visualize")
            return
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        self.plot_wave_in_time_domain(self.wav_signal, "Original Signal", ax=ax[0], color="blue")
        self.plot_wave_in_frequency_domain(self.wav_signal, ax=ax[1], color="blue")
        plt.tight_layout()
        plt.show()
        
    def plot_spectrogram(self, ax=None):
        """Plots the spectrogram of the received WAV signal."""
        if ax is None:
            ax = plt.gca()
            
        
        if self.wav_signal is None:
            print("No signal to plot spectrogram for.")
            return

        # Define hop_length (adjust as needed for desired time resolution)
        hop_length = 256 
        # Calculate the STFT (Short-Time Fourier Transform)
        stft_result = librosa.stft(self.wav_signal.astype(float), hop_length=hop_length)
        # Get the magnitude of the STFT result
        # without decibels
        amplitude_spectrogram = np.abs(stft_result)
        # Convert to decibels (optional, for better visualization)
        amplitude_spectrogram_db = librosa.amplitude_to_db(amplitude_spectrogram, ref=np.max)

        # Display the spectrogram using decibels
        librosa.display.specshow(amplitude_spectrogram_db, sr=config.SAMPLE_RATE, hop_length=hop_length, x_axis="time", y_axis="hz") 
        plt.colorbar(label="Amplitude (dB)") # Updated colorbar label
        ax.set_title(f"Spectrogram (Decibels) for signal with carrier {self.carrier_freq}Hz and bitrate {self.bit_rate}Hz")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, 20000) # Limit y-axis to 20kHz
        
        
    def plot_wave_in_frequency_domain(self, wave, ax=None, color="b", alpha=0.5, label=None):
        """
        Plots the frequency domain representation of the received WAV signal.
        @param wave: The received WAV signal.
        @param ax: The axis to plot on. If None, uses the current axis.
        @param color: The color of the plot line.
        @param alpha: The transparency of the plot line.
        @param label: The label for the plot line for the legend.
        @return: None
        """
        if ax is None:
            ax = plt.gca()

        wave_f = np.fft.fft(wave)
        wave_f = np.fft.fftshift(wave_f)
        frequencies_x_axis = np.arange(
            config.SAMPLE_RATE / -2, config.SAMPLE_RATE / 2, config.SAMPLE_RATE / len(wave)
        )

        frequency_magnitudes = np.abs(wave_f)
        # Make it to decibels
        frequency_magnitudes = 20 * np.log10(frequency_magnitudes / np.max(frequency_magnitudes))

        # only plot the positive frequencies
        positive_frequencies = frequencies_x_axis > 0
        frequencies_x_axis = frequencies_x_axis[positive_frequencies]
        frequency_magnitudes = frequency_magnitudes[positive_frequencies]

        ax.plot(frequencies_x_axis, frequency_magnitudes, "-", color=color, alpha=alpha, label=label) 
        
        ## Make vertical line at bandpass
        ax.axvline(x = self.carrier_freq - (self.bit_rate // 2), color="blue", linestyle="--", label="Bandwidth")
        ax.axvline(x = self.carrier_freq + (self.bit_rate // 2), color="blue", linestyle="--")
        ax.legend(loc='upper right', fontsize='small')
        ax.set_xlim(0, 20000)  # Limit x-axis to 20kHz
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title("Frequency Domain Signal")
        ax.grid(True)

    def plot_spectrogram_and_frequency_domain(self):
        if self.wav_signal is None:
            print("No signal to visualize")
            return

        _ , ax = plt.subplots(2, 1, figsize=(10, 8))
        self.plot_wave_in_frequency_domain(self.wav_signal, ax=ax[0], color="blue")
        self.plot_spectrogram(ax=ax[1])
        plt.tight_layout()
        plt.show()

    def plot_wave_in_time_domain(self, wave, l: str, ax=None, color="orange", alpha=0.5):
        """
        Plots the time domain representation of the received WAV signal.
        @param wave: The received WAV signal.
        @param l: The label for the plot line for the legend.
        @param ax: The axis to plot on. If None, uses the current axis.
        @param color: The color of the plot line.
        @return: None
        """
        if ax is None:
            ax = plt.gca()

        time_array = np.arange(len(wave)) / config.SAMPLE_RATE
        ax.plot(time_array, wave, color=color, label=l, alpha=alpha)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Time Domain Signal")
        ax.grid(True)

    def plot_in_frequency_domain(self):
        """
        Plots raw signal, after envelope, and after lowpass filter in frequency domain.
        @param self: Receiver object
        @return: _type_: None
        """
        if self.wav_signal is None:
            print("No signal to visualize")
            return
        
        _ , debug_info = self.decode()

        fig = create_frequency_domain_visualization(self, debug_info)
        plt.show()
        return fig
    
    def plot_in_frequency_after_bandpass_filter(self):
        if self.wav_signal is None:
            print("No signal to visualize")
            return

        # Apply bandpass filter
        filtered_signal = self.bandpass_filter(self.wav_signal)
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        self.plot_wave_in_frequency_domain(self.wav_signal, ax=ax, color="blue", alpha=0.3)
        self.plot_wave_in_frequency_domain(filtered_signal, ax=ax, color="red", alpha=0.6)
        plt.title("Frequency Domain after Bandpass Filter")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude dB")
        plt.xlim(0, 20000)  # Limit x-axis to 10kHz
        plt.legend(["Original Signal", "After Bandpass Filter Before Envelope"])
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        
    def plot_wave_in_freq_domain_after_envelope(self):
        if self.wav_signal is None:
            print("No signal to visualize")
            return

        # _demodulate returns a tuple, we need the debug_info dictionary
        _filtered_signal, debug_info = self._demodulate() 
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        
        # Plot original signal with label
        self.plot_wave_in_frequency_domain(self.wav_signal, ax=ax, color="blue", alpha=0.3, label="Original Signal")
        # Plot signal after envelope with label
        self.plot_wave_in_frequency_domain(debug_info["envelope"], ax=ax, color="red", alpha=0.6, label="After Envelope before Lowpass Filter")
        # Plot the signal after lowpass filter with label
        self.plot_wave_in_frequency_domain(debug_info["filtered"], ax=ax, color="purple", alpha=0.6, label="After Lowpass Filter")
        
        ### Plot vertical line at bandwidth around carrier frequency
        # Using a distinct color (e.g., green) for Bandwidth lines
        plt.axvline(x=self.carrier_freq - (self.bit_rate // 2), color="blue", linestyle="--", label="Bandwidth")
        plt.axvline(x=self.carrier_freq + (self.bit_rate // 2), color="blue", linestyle="--")
        
        ### Plot vertical line at bit rate
        plt.axvline(x=self.bit_rate // 2, color="red", linestyle="--", label="Bit Rate")
        plt.axvline(x=-self.bit_rate // 2, color="red", linestyle="--")
        
        plt.title("Frequency Domain after Envelope")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude dB")
        plt.xlim(-15000, 15000)  # Limit x-axis
        
        # Add legend, placed in a corner (e.g., 'upper right')
        # This will automatically use the labels provided in plot and axvline calls
        plt.legend(loc='lower left', fontsize='small') 
        
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def plot_wave_in_time_domain_after_envelope(self, msg_original_text:str):
        
        if self.wav_signal is None:
            print("No signal to visualize")
            return

        # _demodulate returns a tuple, we need the debug_info dictionary
        try:
            _, debug_info = self.decode()
        except Exception as e:
            _, debug_info = self._demodulate()

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        
        self.plot_wave_in_time_domain(self.wav_signal, "Original Signal", ax=ax, color="blue", alpha=0.2)
        if "envelope" in debug_info:
            self.plot_wave_in_time_domain(debug_info["envelope"], "After Envelope", ax=ax, color="red", alpha=0.7)
        if "filtered" in debug_info: # This is typically the LPF output on the envelope
            self.plot_wave_in_time_domain(debug_info["filtered"], "After Lowpass Filter", ax=ax, color="purple", alpha=1.0)
        
        # Plot vertical lines for symbol boundaries
        if self.samples_per_symbol > 0 and self.sample_rate > 0:
            for i in range(0, len(self.wav_signal), self.samples_per_symbol):
                # Only plot if within the current xlim to avoid too many lines if zoomed out
                time_sec = i / self.sample_rate
                current_xlim = ax.get_xlim()
                if current_xlim[0] <= time_sec <= current_xlim[1]:
                     plt.axvline(x=time_sec, color="green", linestyle="--", alpha=0.7, linewidth=3.0)
        print(f" values : {debug_info['index_of_9']}")
        ### at debug_info["index_of_9"] plot vertical lines!!
        if "index_of_9" in debug_info:
            for i in range(len(debug_info["index_of_9"])):
                # at index debug_info["index_of_9"][i] plot vertical lines
                time_sec = (debug_info["index_of_9"][i] / self.sample_rate) * self.samples_per_symbol
                current_xlim = ax.get_xlim()
                if current_xlim[0] <= time_sec <= current_xlim[1]:
                    plt.axvline(x=time_sec, color="black", linestyle="--", alpha=0.7, linewidth=3.0)
                    


            
        

        # plt.xlim(1.0, 1.490)  # Limit x-axis to 2 seconds as per user's existing code

        handles, labels = ax.get_legend_handles_labels()
        if handles: # Avoid creating a legend if there's nothing to label
            by_label = dict(zip(labels, handles)) # Remove duplicate labels
            ax.legend(by_label.values(), by_label.keys(), loc='lower left', fontsize='small')
        payload_size = len(msg_original_text) * 8 + 13
        if config.USE_ESP:
            transmitter = "ESP"
        else:
            transmitter = "SG"
        plt.title(f"{transmitter} : Time Domain of signal with payload size {str(payload_size)} bits")
        plt.tight_layout()
        plt.show()
        
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
            # Plot time domain signals
        # fig = self.plot_wave_in_time_domain(self.wav_signal, "Received", color="g")
        plt.show()
        return fig
    def plot_low_pass_filter_bode(self):
        """
        Generates and plots the magnitude response for the low-pass filter.
        Uses a logarithmic x-axis from 1 to 10000 Hz. Font sizes adjusted for A4 paper.
        """
        cutoff_freq_lp = self.cutoff_freq # Hz
        nyquist = 0.5 * config.SAMPLE_RATE
        normalized_cutoff_lp = cutoff_freq_lp / nyquist

        b_lp, a_lp = signal.butter(self.filter_order, normalized_cutoff_lp, btype='low', analog=False)
        
        w_lp, h_lp = signal.freqz(b_lp, a_lp, worN=8000, fs=config.SAMPLE_RATE) 

        mag_lp = 20 * np.log10(np.abs(h_lp))

        # Font sizes for A4 paper
        title_fontsize = 14
        label_fontsize = 14
        legend_fontsize = 14
        tick_fontsize = 14

        plt.figure(figsize=(10, 6)) 
        plt.semilogx(w_lp, mag_lp, color='blue') 
        plt.title(f'Low-Pass Filter Magnitude Response (Order={self.filter_order}, Cutoff={cutoff_freq_lp:.2f} Hz)', fontsize=title_fontsize)
        plt.xlabel('Frequency [Hz] (log scale)', fontsize=label_fontsize)
        plt.ylabel('Magnitude [dB]', fontsize=label_fontsize)
        plt.grid(which='both', linestyle='-', color='grey', alpha=0.5)
        plt.axvline(cutoff_freq_lp, color='red', linestyle='--', label=f'Cutoff Freq: {cutoff_freq_lp:.2f} Hz')
        plt.legend(fontsize=legend_fontsize)
        plt.xlim([1, 10000]) 
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        
        relevant_indices = (w_lp >= 1) & (w_lp <= 10000)
        if np.any(relevant_indices):
            min_mag_plot = np.min(mag_lp[relevant_indices])
            max_mag_plot = np.max(mag_lp[relevant_indices])
            plt.ylim([min_mag_plot - 5, max_mag_plot + 5])
        else: 
            plt.ylim([-100, 10])

        plt.tight_layout()
        plt.show()

    def plot_band_pass_filter_bode(self):
        """
        Generates and plots the magnitude response for the band-pass filter.
        Uses a logarithmic x-axis from 1 to 10000 Hz. Font sizes adjusted for A4 paper.
        """
        low_cutoff_bp = self.carrier_freq - self.bit_rate
        high_cutoff_bp = self.carrier_freq + self.bit_rate
        nyquist = 0.5 * config.SAMPLE_RATE
        normalized_low_bp = low_cutoff_bp / nyquist
        normalized_high_bp = high_cutoff_bp / nyquist

        b_bp, a_bp = signal.butter(self.filter_order, [normalized_low_bp, normalized_high_bp], btype='band', analog=False)
        
        w_bp, h_bp = signal.freqz(b_bp, a_bp, worN=8000, fs=config.SAMPLE_RATE)

        mag_bp = 20 * np.log10(np.abs(h_bp))

        # Font sizes for A4 paper
        title_fontsize = 14
        label_fontsize = 14
        legend_fontsize = 14
        tick_fontsize = 14

        plt.figure(figsize=(10, 6)) 
        plt.semilogx(w_bp, mag_bp, color='green') 
        plt.title(f'Band-Pass Filter Magnitude Response (Order={self.filter_order}, Passband: {low_cutoff_bp}-{high_cutoff_bp} Hz)', fontsize=title_fontsize)
        plt.xlabel('Frequency [Hz] (log scale)', fontsize=label_fontsize)
        plt.ylabel('Magnitude [dB]', fontsize=label_fontsize)
        plt.grid(which='both', linestyle='-', color='grey', alpha=0.5)
        plt.axvline(low_cutoff_bp, color='red', linestyle='--', label=f'Low Cutoff: {low_cutoff_bp} Hz')
        plt.axvline(high_cutoff_bp, color='orange', linestyle='--', label=f'High Cutoff: {high_cutoff_bp} Hz')
        plt.axvline(self.carrier_freq, color='purple', linestyle=':', label=f'Center Freq: {self.carrier_freq} Hz')
        plt.legend(fontsize=legend_fontsize)
        plt.xlim([1, 10000]) 
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)

        relevant_indices = (w_bp >= 1) & (w_bp <= 10000)
        if np.any(relevant_indices):
            min_mag_plot = np.min(mag_bp[relevant_indices])
            max_mag_plot = np.max(mag_bp[relevant_indices])
            plt.ylim([min_mag_plot - 5, max_mag_plot + 5])
        else: 
            plt.ylim([-100, 10])

        plt.tight_layout()
        plt.show()
    
    
    # def plot_envelope_thresholded_with_samples_per_symbol_ticls(self):

    
    
    
    
    
    
    
    
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

        if APPLY_BAKER_PREAMBLE:
            bits_without_preamble, all_data_bits, index_of_9 = self.remove_preamble_baker_code(bits)

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
            "all_data_bits": all_data_bits if APPLY_BAKER_PREAMBLE else None,
            "normalized": normalized,
            "thresholded": thresholded,
            "bits_without_preamble": bits_without_preamble,
            "index_of_9": index_of_9 if APPLY_BAKER_PREAMBLE else None,
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
    hilbert_receiver = NonCoherentReceiver.from_wav_file(f"Code/dsp/data/raw_data/{IS_ID_SPECIFIED}.wav")
    message, debug = hilbert_receiver.decode()
    print("Hilbert Decoded:", message)
    hilbert_receiver.plot_simulation_steps()

    # Shift-Imaginary method
    shift_receiver = CoherentReceiver.from_wav_file(PATH_TO_WAV_FILE)
    message, debug = shift_receiver.decode()
    print("Shift Decoded:", message)
    shift_receiver.plot_simulation_steps()
