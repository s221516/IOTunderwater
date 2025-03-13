import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import time


from encoding.convolutional_encoding import conv_encode, bit_string_to_list, list_to_bit_string
import config
from Code.dsp.config import (
    PATH_TO_WAV_FILE,
    SAMPLE_RATE,
    SAMPLE_RATE_FOR_WAV_FILE,
    CONVOLUTIONAL_CODING,
    MESSAGE,
    PORT
)

np.set_printoptions(precision=4, suppress=True)

def make_square_wave(message):
    """Convert the message into a binary square wave."""
    message_binary = "".join(format(ord(i), "08b") for i in message)
    
    if (CONVOLUTIONAL_CODING):
        message_binary = bit_string_to_list(message_binary)
        message_binary = conv_encode(message_binary)
        message_binary = list_to_bit_string(message_binary)

    print("Message binary:", message_binary)
    square_wave = []
    for bit in message_binary:
        square_wave += [int(bit)] * config.SAMPLES_PER_SYMBOL
    # Ensure even sampling alignment
    if len(square_wave) % config.SAMPLES_PER_SYMBOL != 0:
        square_wave += [0] * (
            config.SAMPLES_PER_SYMBOL - len(square_wave) % config.SAMPLES_PER_SYMBOL
        )
    duration = len(square_wave) / SAMPLE_RATE
    time_array = np.linspace(0, duration, len(square_wave))
    square_wave = np.array(square_wave)

    return square_wave, time_array

def make_carrier_wave(time_array):
    """Generate the carrier wave."""
    carrier_wave = np.sin(2 * np.pi * config.CARRIER_FREQ * time_array)
    return carrier_wave

def create_modulated_wave(square_wave, carrier_wave):
    """Modulate the square wave with the carrier wave."""
    if square_wave is None or carrier_wave is None:
        raise ValueError("Square wave or carrier wave not initialized.")
    # For AM, we need to ensure the square wave is offset to be positive
    normalized_square_wave = 0.5 + 0.5 * square_wave
    
    return normalized_square_wave * carrier_wave 

def write_to_wav_file(modulated_wave):
    """Write the modulated wave to a WAV file."""
    wav.write(PATH_TO_WAV_FILE, SAMPLE_RATE_FOR_WAV_FILE, modulated_wave)

def plot_waveforms(square_wave, carrier_wave, modulated_wave, time_array):
    """Plot the square wave, carrier wave, and modulated wave."""
    if (
        square_wave is None
        or carrier_wave is None
        or modulated_wave is None
    ):
        raise ValueError("Waveforms not fully initialized.")
    plt.figure(figsize=(12, 6))

    # Plot the square wave
    plt.subplot(3, 1, 1)
    plt.plot(time_array, square_wave, label="Square Wave")
    plt.title("Square Wave")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # Plot the carrier wave
    plt.subplot(3, 1, 2)
    plt.plot(time_array, carrier_wave, label="Carrier Wave", color="orange")
    plt.title("Carrier Wave")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # Plot the modulated wave
    plt.subplot(3, 1, 3)
    plt.plot(time_array, modulated_wave, label="Modulated Wave", color="pink")
    plt.title("Modulated Wave")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()



def transmitVirtual(message=MESSAGE):
    """Run virtual transmission process - send to wav.file."""
    square_wave, time_array = make_square_wave(message)
    carrier_wave = make_carrier_wave(time_array)
    modulated_wave = create_modulated_wave(square_wave, carrier_wave)
    write_to_wav_file(modulated_wave)
    #plot_waveforms(square_wave, carrier_wave, modulated_wave, time_array)




    
    

