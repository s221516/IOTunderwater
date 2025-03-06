import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from config_values import MESSAGE, SAMPLES_PER_SYMBOL, SAMPLE_RATE, CARRIER_FREQ, PATH_TO_WAV_FILE 
from plots import plot_TRANSMITTER
np.set_printoptions(precision=4, suppress=True)

def make_square_wave(message: str): #ON OFF KEYING
    message_binary = ''.join(format(ord(i), '08b') for i in message)
    print(f"Message in binary: {message_binary}")
    # TODO: determine the exact number of samples per bit that makes sense in relation to our sample rate 
    # and bits per second and also how this is done in the signal generator
    
    square_wave = []
    for bit in message_binary:
        square_wave += [int(bit)] * SAMPLES_PER_SYMBOL

    duration_of_wav_signal = len(square_wave) / SAMPLE_RATE

    time_array = np.linspace(0, duration_of_wav_signal, len(square_wave))
    square_wave = np.array(square_wave)

    return square_wave, time_array

def make_carrier_wave(time_array) -> np.array:
    # NOTE: to add noise check the bottom of freq domain section of pysdr
    carrier_wave = np.sin(2 * np.pi * CARRIER_FREQ * time_array) 
    
    return carrier_wave

def create_modulated_wave(square_wave: np.array, carrier_wave: np.array) -> np.array:
    return square_wave * carrier_wave

def save_wave_file(modulated_wave):
    wav.write(PATH_TO_WAV_FILE, SAMPLE_RATE, modulated_wave)
    

if __name__ == "__main__":
    print("COCK")
    square_wave, time_array = make_square_wave(MESSAGE)
    carrier_wave = make_carrier_wave(time_array)
    modulated_wave = create_modulated_wave(square_wave, carrier_wave)
    save_wave_file(modulated_wave)

    print("trans", modulated_wave)
    plot_TRANSMITTER(square_wave, carrier_wave, modulated_wave, time_array)