import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from image_decoding import *
import cv2

# nyquist = 30400
# 31650 / 15200
SAMPLE_RATE = 50000
CARRIER_FREQ = 15200  #  15200 Hz
BIT_RATE = 1000  # 1 Khz
ACTIVATION_ENERGY_THRESHOLD = 0.4 # lower bound for start of 
# with 200 samples per bit and 0.45 noise amplitude -> about upper limit, we get a valid picture 1/3 times
NOISE_AMPLITUDE = 0.3 # noise
PATH_TO_WAV_FILE = "Code/dsp/data/signal.wav"
SAMPLE_RATE_FOR_WAV_FILE = 44100 # Hz

def butter_lowpass(cutoff, freq_sampling, order):
    nyquist = 0.5 * freq_sampling
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, freq_sampling, order):
    b, a = butter_lowpass(cutoff, freq_sampling, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def read_wavefile():
    data_from_wav_file = wav.read(PATH_TO_WAV_FILE)
    freq_sample = data_from_wav_file[0]
    signal = data_from_wav_file[1] / 32767.0
    return freq_sample, signal  

def encode_and_modulate(message):
    """Encode text message and create AM modulated signal"""
    # Convert text to binary
    binary_message = ""
    for c in message:
        byte = format(ord(c), "08b")
        binary_message += byte

    # print(f"Binary message: {binary_message}")

    # Calculate timing parameters
    duration_per_bit = 1 / BIT_RATE
    
    samples_per_bit = int(SAMPLE_RATE * duration_per_bit)
    # Create baseband signal with oversampling
    bits = np.array([int(b) for b in binary_message])
    baseband = np.repeat(bits, samples_per_bit)

    # Add padding at start and end
    padding = np.zeros(samples_per_bit * 10)
    data_wave = np.concatenate([padding, baseband, padding])

    ## TODO: understand pulse shaping and whether we need it
    # # Apply pulse shaping (raised cosine)
    # alpha = 0.35  # Roll-off factor
    # symbol_length = samples_per_bit
    # time_array = np.arange(-4 * symbol_length, 4 * symbol_length)
    # h = (
    #     np.sinc(time_array / symbol_length)
    #     * np.cos(np.pi * alpha * time_array / symbol_length)
    #     / (1 - (2 * alpha * time_array / symbol_length) ** 2)
    # )
    # h[np.isnan(h)] = 1  # Fix division by zero
    # h = h / np.sum(h)  # Normalize

    # # Apply pulse shaping
    # data_wave = np.convolve(baseband, h, "same")
   
    # Scale to ensure good modulation depth (0.2 to 1.0)
    data_wave = 0.2 + 0.85 * (data_wave - np.min(data_wave)) / (np.max(data_wave) - np.min(data_wave))

    # Generate time array
    time_array = np.arange(len(data_wave)) / SAMPLE_RATE 

    # After (direct frequency usage, ensure carrier_freq < sample_rate/2):
    carrier = np.sin(2 * np.pi * CARRIER_FREQ * time_array)

    # Modulate
    modulated = data_wave * carrier

    # Add Gaussion noise, find out if this is the correct interval, maybe its -x to x and not 0 to x
    noise = np.random.normal(0, NOISE_AMPLITUDE, len(modulated))
    modulated_with_noise = modulated + noise

    modulated_with_noise = (modulated_with_noise * 32767).astype(np.int16)  # Convert to 16-bit PCM format

    # Compute SNR and Shannon
    # snr, shannon_limit = compute_snr_and_shannon_limit(modulated, noise)
    # print(f"Signal-to-noise ratio: {snr} dB")
    # print(f"Shannon limit: {shannon_limit}")

    # Write to wave file, with specificed sampling rate and data array
    wav.write(PATH_TO_WAV_FILE, SAMPLE_RATE_FOR_WAV_FILE, modulated_with_noise)

    return modulated_with_noise, time_array

def demodulate_and_decode(modulated):
    """Demodulate AM signal and decode message"""
    # NOTE: If this simple envelope calculation is not enough, then consider hilbert transform
    envelope = np.abs(modulated)

    cutoff = BIT_RATE
    signal_post_filter = butter_lowpass_filter(envelope, cutoff, SAMPLE_RATE, order=4)

    signal_post_filter = signal_post_filter - np.mean(signal_post_filter)

    s_min = np.min(signal_post_filter)
    s_max = np.max(signal_post_filter)
    error = 1e-12

    normalized = (signal_post_filter - s_min) / (s_max - s_min + error)

    # Calculate samples per bit
    samples_per_bit = int(SAMPLE_RATE / BIT_RATE)

    # Create a matched filter for bit detection
    matched_filter = np.ones(samples_per_bit)

    # TODO: understand why it is necessary to convolve here
    # Find start of data by looking for first significant transition
    energy = np.convolve(normalized, matched_filter, "valid")

    # maybe add a noise filter here, but by slightly lowering the max energy required it got super clear
    start_of_valid_data_array = np.where(energy > ACTIVATION_ENERGY_THRESHOLD * np.max(energy))[0]
    start_index = start_of_valid_data_array[0]

    # makes the bitstring for the valid data array
    bits = []
    for i in range(start_index, len(normalized) - samples_per_bit, samples_per_bit):
        bit_sample = normalized[i : i + samples_per_bit]
        bit_value = 1 if np.sum(bit_sample) > (samples_per_bit / 2) else 0
        bits.append(bit_value)

    # Convert bits to ASCII (8 bits per character)
    message = ""
    for i in range(0, len(bits) - 7, 8):
        char_bits = bits[i : i + 8]
        if len(char_bits) == 8:
            char_code = int("".join(map(str, char_bits)), 2)
            # print(f"Bits: {char_bits}, Char Code: {char_code}, Char: {chr(char_code) if 32 <= char_code <= 126 else 'Non-printable'}")
            if 32 <= char_code <= 126:  # Printable ASCII
                message += chr(char_code)

    return message, normalized, energy, bits

def plot_debug(t, modulated, envelope, bits, energy, samples_to_plot=None):
    """Create debug plots with signal information"""
    if samples_to_plot is None:
        samples_to_plot = len(t)

    plt.figure(figsize=(15, 12))

    # Plot modulated signal
    plt.subplot(4, 1, 1)
    plt.plot(t[:samples_to_plot], modulated[:samples_to_plot])
    plt.title(
        f"Modulated Signal (max={np.max(modulated):.2f}, min={np.min(modulated):.2f})"
    )
    plt.grid(True)

    # Plot envelope
    plt.subplot(4, 1, 2)
    plt.plot(t[:samples_to_plot], envelope[:samples_to_plot])
    plt.title(f"Envelope (max={np.max(envelope):.2f}, min={np.min(envelope):.2f})")
    plt.grid(True)

    # Plot energy
    plt.subplot(4, 1, 3)
    plt.plot(energy[:samples_to_plot])
    plt.title("Energy plot")
    plt.grid(True)

    # Plot bits if available
    # if len(bits) > 0:
    #     plt.subplot(4, 1, 4)
    #     plt.step(range(len(bits)), bits, where="post")
    #     plt.title(f"Decoded Bits (total: {len(bits)})")
    #     plt.grid(True)

    plt.tight_layout()
    plt.show()

def compute_snr_and_shannon_limit(signal, noise):
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10((signal_power - noise_power) / noise_power)
    B = BIT_RATE
    S = np.mean(modulated**2)
    N = np.mean(noise**2)
    shannon_limit = B * np.log2(1 + (S / N))
    return snr, shannon_limit

if __name__ == "__main__":
    with open("Code/dsp/picture_in_binary.txt", "r") as file:
        picture_in_binary = file.read()

    all_letters = "the quick brown fox jumps over the lazy dog while vexd zebras fight for joy! @#$%^&()_+[]{}|;:,.<>/?~` \ The 5 big oxen love quick daft zebras & dogs.>*"
    small_test = "This is: 14"
    picture_in_binary_with_prefix = "p" + picture_in_binary
    text_with_prefix = "t" + small_test

    MESSAGE = picture_in_binary_with_prefix

    print("Encoding message...")
    modulated, time_array = encode_and_modulate(MESSAGE)

    print("Reading wav file...")
    freq_sample, signal_from_wave_file = read_wavefile()

    print("Demodulating message...")
    decoded_message, envelope, bits, energy = demodulate_and_decode(signal_from_wave_file)
    print(f"Decoded message: {decoded_message[1:]}")

    pixels = convert_bits_to_image(decoded_message[1:])
    show_picture(pixels)

    # prefix_from_message = decoded_message[0]
    # if (prefix_from_message == "p"):

    # if (prefix_from_message == "t"):
    #     print(f"Original message: {MESSAGE}")
    

    samples_to_plot = int( 0.1 * SAMPLE_RATE)  
    # plot_debug(time_array, modulated, envelope, bits, energy, samples_to_plot)
  