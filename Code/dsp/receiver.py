import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
import scipy.signal as signal

from config_values import PATH_TO_WAV_FILE, BIT_RATE, SAMPLE_RATE, ACTIVATION_ENERGY_THRESHOLD, CARRIER_FREQ


def read_wavefile():
    data_from_wav_file = wav.read(PATH_TO_WAV_FILE)
    freq_sample = data_from_wav_file[0]
    signal = data_from_wav_file[1] / 32767.0
    return freq_sample, signal  


def butter_lowpass(cutoff, freq_sampling, order):
    nyquist = 0.5 * freq_sampling
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, freq_sampling, order):
    b, a = butter_lowpass(cutoff, freq_sampling, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def demodulate_and_decode(modulated):
    """Demodulate AM signal and decode message"""
    # NOTE: If this simple envelope calculation is not enough, then consider hilbert transform
    # TODO: add windowing i.e. blackman or hamming
    envelope = np.abs(modulated)

    cutoff = (CARRIER_FREQ - BIT_RATE) / 2
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


# def compute_snr_and_shannon_limit(signal, noise):
#     signal_power = np.mean(signal**2)
#     noise_power = np.mean(noise**2)
#     snr = 10 * np.log10((signal_power - noise_power) / noise_power)
#     B = BIT_RATE
#     S = np.mean(modulated**2)
#     N = np.mean(noise**2)
#     shannon_limit = B * np.log2(1 + (S / N))
#     return snr, shannon_limit
