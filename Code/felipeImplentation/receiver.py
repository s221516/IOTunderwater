import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from config_values import *
from plots import *
np.set_printoptions(precision=4, suppress=True, linewidth=120)

def read_wave_file(file_path):
    sample_rate, modulated_wave = wav.read(file_path)
    return modulated_wave, sample_rate

def rectify_signal(signal):
    rectified_signal = np.abs(signal)
    return rectified_signal
    
def low_pass_filter(rectified_signal, sample_rate):

    nyquist = 0.5 * sample_rate #The Nyquist frequency is half the sample rate and defines the highest frequency that can be represented in a digital signal without aliasing
    normal_cutoff = CUTOFF / nyquist # normal_cutoff = 1, means the cutoff is at max frequency.

    #TODO understand these:
    b, a = butter(ORDER, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, rectified_signal)

def decode(filtered_signal, sample_rate):
    decoded_bits = []

    chuncks = np.arange(0, len(filtered_signal), SAMPLES_PER_SYMBOL)
    for i in chuncks:
        
        symbol = filtered_signal[i:i + SAMPLES_PER_SYMBOL] #check entry 0-16, then 16-32 etc.
        avg_value = np.mean(symbol)
    
        if avg_value > 0.5: 
            decoded_bits.append(1)
        else:
            decoded_bits.append(0)
    
    decoded_message = ''.join(str(bit) for bit in decoded_bits)
    return decoded_message


def binary_to_ascii(binary_str):
    # Split the binary string into 8-bit chunks
    binary_values = [binary_str[i:i+8] for i in range(0, len(binary_str), 8)]
    
    # Convert each binary value to its ASCII character
    ascii_chars = [chr(int(bv, 2)) for bv in binary_values]
    
    # Join the list of ASCII characters into a string
    ascii_str = ''.join(ascii_chars)
    
    return ascii_str



if __name__ == "__main__":
    modulated_wave, sample_rate = read_wave_file(PATH_TO_WAV_FILE)
    rectified_signal = rectify_signal(modulated_wave)
    filtered_signal = low_pass_filter(rectified_signal, sample_rate)
    decoded = decode(filtered_signal, sample_rate)

    print(binary_to_ascii(decoded))

    visualize_transformations(
    [modulated_wave, rectified_signal, filtered_signal],
    sample_rate,
    ["Original Modulated Wave", "Rectified Signal", "Filtered (Envelope) Signal"]
)


