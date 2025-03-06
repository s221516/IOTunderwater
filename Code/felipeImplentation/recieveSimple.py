import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from config_values import *

np.set_printoptions(precision=4, suppress=True, linewidth=120)

def read_wave_file(file_path):
    sample_rate, modulated_wave = wav.read(file_path)
    return modulated_wave, sample_rate

def rectify_signal(signal):
    rectified_signal = np.abs(signal)
    return rectified_signal
    
def low_pass_filter(rectified_signal, sample_rate):

    nyquist = 0.5 * sample_rate #The Nyquist frequency is half the sample rate and defines the highest frequency that can be represented in a digital signal without aliasing
    normal_cutoff = 15000 / nyquist # normal_cutoff = 1, means the cutoff is at max frequency.

    #TODO understand these:
    b, a = butter(5, normal_cutoff, btype='low', analog=False)
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

def plot_waveform(modulated_wave, sample_rate):
    """Plot the waveform from the modulated wave."""
    # Generate time axis
    time_array = np.linspace(0, len(modulated_wave) / sample_rate, num=len(modulated_wave))

    # Plot the waveform with a specific color
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, modulated_wave, color='blue')  # Specify the color here
    plt.title("Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()



def recieve():
    modulated_wave, sample_rate = read_wave_file(PATH_TO_WAV_FILE)
    plot_waveform(modulated_wave, sample_rate)
    rectified_signal = rectify_signal(modulated_wave)
    filtered_signal = low_pass_filter(rectified_signal, sample_rate)
    decoded = decode(filtered_signal, sample_rate)


