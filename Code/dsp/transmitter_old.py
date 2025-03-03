import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
from config_values import BIT_RATE, SAMPLES_PER_SYMBOL, SAMPLE_RATE, CARRIER_FREQ, NOISE_AMPLITUDE, PATH_TO_WAV_FILE, SAMPLE_RATE_FOR_WAV_FILE
np.set_printoptions(precision=8, suppress=True)

def encode_and_modulate(message):
    """Encode text message and create AM modulated signal"""
    
    # 2. Construct square wave signal from message
    def create_square_wave_signal(message):
        # convert message to binary
        binary_message = ''.join(format(ord(char), '08b') for char in message)
        
        square_wave_signal = []
        
        for b in binary_message:
            # TODO: probably too high of a samples per bit
            square_wave_signal += [int(b)] * SAMPLES_PER_SYMBOL
        
        return square_wave_signal
    
    # 1. Construct a carrier signal
    def create_carrier_signal(size_of_data, carrier=CARRIER_FREQ):
        
        duration = size_of_data / SAMPLE_RATE
        time_array = np.linspace(0, duration, size_of_data)
        print(duration, len(time_array))
        return np.sin(2 * np.pi * carrier * time_array)     
    
    # 3. Amplitude Modulate the square wave signal with the carrier signal
    def combine_square_wave_and_carrier(square_wave_signal, carrier_signal):
        return square_wave_signal * carrier_signal
    
    square_wave = create_square_wave_signal(message)
    size_of_data = len(square_wave)
    carrier     = create_carrier_signal(size_of_data)
    am_modulated_signal = combine_square_wave_and_carrier(square_wave, carrier)
    # print(len(time_array), np.shape(time_array))
    
    plt.figure(figsize=(20, 20))

    time_array = np.linspace(0, size_of_data / BIT_RATE, size_of_data)
    
    plt.subplot(3, 1, 1)
    plt.plot(time_array, square_wave)
    plt.title("Square wave signal")
    
    plt.subplot(3, 1, 2)
    plt.plot(time_array, carrier)
    plt.title("Carrier signal")
    
    plt.subplot(3, 1, 3)
    plt.plot(time_array, am_modulated_signal)
    plt.title("AM Modulated signal")
    
    plt.show()
    
    
    return am_modulated_signal

if __name__ == "__main__":
    encode_and_modulate("h")