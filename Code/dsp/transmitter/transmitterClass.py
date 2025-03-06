import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
from encoding.convolutional_encoding import conv_encode, bit_string_to_list, list_to_bit_string
from config_values import (
    CARRIER_FREQ,
    PATH_TO_WAV_FILE,
    SAMPLE_RATE,
    SAMPLE_RATE_FOR_WAV_FILE,
    SAMPLES_PER_SYMBOL,
    CONVOLUTIONAL_CODING
)

np.set_printoptions(precision=4, suppress=True)

class Transmitter:
    def __init__(self, message: str):
        self.message = message
        self.square_wave = None
        self.time_array = None
        self.carrier_wave = None
        self.modulated_wave = None

    def make_square_wave(self):
        """Convert the message into a binary square wave."""
        message_binary = "".join(format(ord(i), "08b") for i in self.message)
        
        if (CONVOLUTIONAL_CODING):

            message_binary = bit_string_to_list(message_binary)
            message_binary = conv_encode(message_binary)
            message_binary = list_to_bit_string(message_binary)

        print("Message binary:", message_binary)
        square_wave = []
        for bit in message_binary:
            square_wave += [int(bit)] * SAMPLES_PER_SYMBOL
        # Ensure even sampling alignment
        if len(square_wave) % SAMPLES_PER_SYMBOL != 0:
            square_wave += [0] * (
                SAMPLES_PER_SYMBOL - len(square_wave) % SAMPLES_PER_SYMBOL
            )
        duration = len(square_wave) / SAMPLE_RATE
        time_array = np.linspace(0, duration, len(square_wave))
        self.square_wave = np.array(square_wave)
        self.time_array = time_array

    def make_carrier_wave(self):
        """Generate the carrier wave."""
        if self.time_array is None:
            raise ValueError("Time array not initialized. Call make_square_wave first.")
        self.carrier_wave = np.sin(2 * np.pi * CARRIER_FREQ * self.time_array)

    def create_modulated_wave(self):
        """Modulate the square wave with the carrier wave."""
        if self.square_wave is None or self.carrier_wave is None:
            raise ValueError("Square wave or carrier wave not initialized.")
        # For AM, we need to ensure the square wave is offset to be positive
        normalized_square_wave = 0.5 + 0.5 * self.square_wave
        self.modulated_wave = normalized_square_wave * self.carrier_wave

    def write_to_wav_file(self):
        """Write the modulated wave to a WAV file."""
        if self.modulated_wave is None:
            raise ValueError("Modulated wave not initialized.")
        wav.write(PATH_TO_WAV_FILE, SAMPLE_RATE_FOR_WAV_FILE, self.modulated_wave)

    def plot_waveforms(self):
        """Plot the square wave, carrier wave, and modulated wave."""
        if (
            self.square_wave is None
            or self.carrier_wave is None
            or self.modulated_wave is None
        ):
            raise ValueError("Waveforms not fully initialized.")
        plt.figure(figsize=(12, 6))

        # Plot the square wave
        plt.subplot(3, 1, 1)
        plt.plot(self.time_array, self.square_wave, label="Square Wave")
        plt.title("Square Wave")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        # Plot the carrier wave
        plt.subplot(3, 1, 2)
        plt.plot(self.time_array, self.carrier_wave, label="Carrier Wave", color="orange")
        plt.title("Carrier Wave")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        # Plot the modulated wave
        plt.subplot(3, 1, 3)
        plt.plot(self.time_array, self.modulated_wave, label="Modulated Wave", color="pink")
        plt.title("Modulated Wave")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def transmit(self):
        """Run the full transmission process."""
        self.make_square_wave()
        self.make_carrier_wave()
        self.create_modulated_wave()
        self.write_to_wav_file()
        self.plot_waveforms()