import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from config_values import BIT_RATE, SAMPLES_PER_SYMBOL, SAMPLE_RATE, CARRIER_FREQ, NOISE_AMPLITUDE, PATH_TO_WAV_FILE, SAMPLE_RATE_FOR_WAV_FILE
np.set_printoptions(precision=4, suppress=True)

# returns two arrays, square_wave and the time_array
def make_square_wave(message: str):
    message_binary = ''.join(format(ord(i), '08b') for i in message)
    print(f"Message in binary: {message_binary}")
    # TODO: determine the exact number of samples per bit that makes sense in relation to our sample rate 
    # and bits per second and also how this is done in the signal generator
    
    square_wave = []
    for bit in message_binary:
        square_wave += [int(bit)] * SAMPLES_PER_SYMBOL

    duration = len(square_wave) / SAMPLE_RATE

    time_array = np.linspace(0, duration, len(square_wave))
    square_wave = np.array(square_wave)

    return square_wave, time_array

def make_carrier_wave(time_array) -> np.array:
    # NOTE: to add noise check the bottom of freq domain section of pysdr
    carrier_wave = np.sin(2 * np.pi * CARRIER_FREQ * time_array) 
    
    return carrier_wave

def plot_waveforms(square_wave, carrier_wave, modulated_wave, time_array):
    plt.figure(figsize=(12, 6))

    # Plot the square wave
    plt.subplot(3, 1, 1)
    plt.plot(time_array, square_wave, label='Square Wave')
    plt.title('Square Wave')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # Plot the carrier wave
    plt.subplot(3, 1, 2)
    plt.plot(time_array, carrier_wave, label='Carrier Wave', color='orange')
    plt.title('Carrier Wave')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # Plot the carrier wave
    plt.subplot(3, 1, 3)
    plt.plot(time_array, modulated_wave, label='Modulated Wave', color='pink')
    plt.title('Modulated Wave')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def create_modulated_wave(square_wave: np.array, carrier_wave: np.array) -> np.array:
    return square_wave * carrier_wave

if __name__ == "__main__":
    square_wave, time_array = make_square_wave("h")
    carrier_wave = make_carrier_wave(time_array)
    print(len(time_array), np.shape(time_array))
    modulated_wave = create_modulated_wave(square_wave, carrier_wave)
    plot_waveforms(square_wave, carrier_wave, modulated_wave, time_array)