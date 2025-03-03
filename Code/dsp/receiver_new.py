import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
from config_values import PATH_TO_WAV_FILE, SAMPLE_RATE, CARRIER_FREQ, CUT_OFF_FREQ, SAMPLES_PER_SYMBOL
def get_wav_signal(path : str):
    _, wav_signal = wavfile.read(path)
    wav_signal    = wav_signal / 32767.0
    return wav_signal

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
    
def normalize_signal(mean_shifted_signal):
    """Normalization values from [-32767, 32676] --> [0,1]"""
    
    normalized_signal = (mean_shifted_signal - np.min(mean_shifted_signal)) / (
        np.max(mean_shifted_signal) - np.min(mean_shifted_signal)
    )
    return normalized_signal
    
def threshold_signal(normalized_signal):
    thresholded_signal = np.where(normalized_signal > 0.5, 1, 0)
    return thresholded_signal

def decode_message(thresholded_signal):

    def find_start_index(thresholded_signal):
        for i in range(len(thresholded_signal)):
            if thresholded_signal[i] != thresholded_signal[i - 1]:
                return i + SAMPLES_PER_SYMBOL // 2
        return -1

    def get_bits_from_thresholded_signal(thresholded_signal):

        bits = []
        start_index = find_start_index(thresholded_signal)
        if start_index == -1:
            return print("No start index found")

        max_index = len(thresholded_signal) - SAMPLES_PER_SYMBOL

        for i in range(start_index, max_index, SAMPLES_PER_SYMBOL):
            bit_sample = thresholded_signal[i : i + SAMPLES_PER_SYMBOL]
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

def decode_wav_signal(wav_signal):
    # envelope = np.abs((signal.hilbert(wav_signal))
    wav_signal              = np.abs(signal.hilbert(wav_signal))
    wav_signal_shifted      = shift_signal(wav_signal)
    filtered_signal         = filter_signal(wav_signal_shifted)
    mean_shifted_signal     = filtered_signal - np.mean(filtered_signal)
    normalized_signal       = normalize_signal(mean_shifted_signal)
    tresholded_signal       = threshold_signal(normalized_signal)
    decoded_message         = decode_message(tresholded_signal)
    
    return decoded_message

if __name__ == '__main__':
    wav_signal          = get_wav_signal(PATH_TO_WAV_FILE)
    decoded_wav_signal  = decode_wav_signal(wav_signal)
    time_array = np.linspace(0, (len(decoded_wav_signal) / SAMPLE_RATE), len(decoded_wav_signal))
    print("DEBUGGING")
    print(decoded_wav_signal, len(decoded_wav_signal))
