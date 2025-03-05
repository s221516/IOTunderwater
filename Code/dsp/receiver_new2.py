import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from config_values import (
    CARRIER_FREQ,
    CUT_OFF_FREQ,
    PATH_TO_WAV_FILE,
    SAMPLE_RATE,
    SAMPLES_PER_SYMBOL,
)
from scipy.io import wavfile


def get_wav_signal(path: str):
    _, wav_signal = wavfile.read(path)
    wav_signal = wav_signal / 32767.0
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
    thresholded_signal = np.where(normalized_signal > 0.55, 1, 0)
    return thresholded_signal


def decode_message(thresholded_signal):
    """
    Convert the thresholded signal into a message by:
    1. Finding where the bit transitions start
    2. Sampling bits at regular intervals based on SAMPLES_PER_SYMBOL
    3. Converting bit groups into ASCII characters
    """
    bits = get_bits_from_thresholded_signal(thresholded_signal)
    if not bits:
        return "No valid bits found"

    message = get_message_from_bits(bits)
    return message


def get_bits_from_thresholded_signal(thresholded_signal):
    """
    Extract bits from the thresholded signal with improved noise handling
    """
    bits = []
    start_index = find_start_index(thresholded_signal)

    if start_index == -1:
        print("Warning: No bit transitions found in signal")
        return []

    max_index = len(thresholded_signal) - SAMPLES_PER_SYMBOL

    # Debug: Print some useful info
    print(f"Start index: {start_index}")
    print(f"Samples per symbol: {SAMPLES_PER_SYMBOL}")
    print(f"Total samples: {len(thresholded_signal)}")
    print(f"Expected bits: ~{(max_index - start_index) // SAMPLES_PER_SYMBOL}")

    for i in range(start_index, max_index, SAMPLES_PER_SYMBOL):
        # Get all samples for this bit
        bit_samples = thresholded_signal[i : i + SAMPLES_PER_SYMBOL]

        # Use majority voting for noise resistance
        ones_count = np.sum(bit_samples)
        zeros_count = len(bit_samples) - ones_count

        if ones_count > zeros_count:
            bits.append(1)
        else:
            bits.append(0)

    return bits


def get_message_from_bits(bits):
    """
    Convert bits to ASCII characters with improved error handling
    """
    message = ""
    # Print raw bits for debugging
    print(f"Raw bits: {''.join(map(str, bits))}")

    # Try to align to byte boundaries if we might have missed the start
    # by trying different offsets
    best_message = ""
    best_printable_count = 0

    for offset in range(8):
        if offset >= len(bits):
            break

        test_message = ""
        printable_count = 0

        for i in range(offset, len(bits) - 7, 8):
            char_bits = bits[i : i + 8]
            if len(char_bits) == 8:
                char_code = int("".join(map(str, char_bits)), 2)
                char = chr(char_code)
                test_message += char

                # Count printable characters
                if 32 <= char_code <= 126:
                    printable_count += 1

        # Keep track of which offset gives the most printable characters
        if printable_count > best_printable_count:
            best_printable_count = printable_count
            best_message = test_message

    # Return the best message we found
    if best_printable_count > 0:
        return best_message
    else:
        return "No valid ASCII characters found"


def find_start_index(thresholded_signal):
    """Find the index where the first bit transition happens"""
    for i in range(1, len(thresholded_signal)):
        if thresholded_signal[i] != thresholded_signal[i - 1]:
            return i + SAMPLES_PER_SYMBOL // 2
    return -1


def coherent_demodulate(wav_signal):
    """
    Perform coherent (synchronous) demodulation of an AM signal:
    1. Multiply the signal by a local oscillator at the carrier frequency
    2. Low-pass filter to extract the baseband signal
    3. Process the resulting signal to recover the digital message
    """
    # Create time array for the local oscillator
    duration_of_wav_signal = len(wav_signal) / SAMPLE_RATE
    time_array = np.linspace(0, duration_of_wav_signal, len(wav_signal))

    # Create local oscillator at carrier frequency
    # For coherent demodulation, we need BOTH cosine and sine components
    # (called I and Q channels in communications theory)
    i_local_osc = np.cos(2 * np.pi * CARRIER_FREQ * time_array)
    q_local_osc = np.sin(2 * np.pi * CARRIER_FREQ * time_array)

    # Multiply the received signal with both local oscillators
    i_mixed = wav_signal * i_local_osc  # In-phase component
    q_mixed = wav_signal * q_local_osc  # Quadrature component

    # Low-pass filter both I and Q components to remove the high-frequency terms
    i_demodulated = filter_signal(i_mixed)
    q_demodulated = filter_signal(q_mixed)

    # For AM, we can use just the I component if phase is aligned
    # But for robustness, we can combine I and Q (essentially doing envelope detection of the baseband signal)
    demodulated_signal = np.sqrt(i_demodulated**2 + q_demodulated**2)

    # Alternatively, if you're sure about phase synchronization:
    # demodulated_signal = i_demodulated

    # Remove DC offset
    mean_shifted_signal = demodulated_signal - np.mean(demodulated_signal)

    # Normalize to [0,1] range
    normalized_signal = normalize_signal(mean_shifted_signal)

    # Apply threshold to recover digital bits
    thresholded_signal = threshold_signal(normalized_signal)

    # Decode the binary message
    decoded_message = decode_message(thresholded_signal)

    return decoded_message, {
        "i_mixed": i_mixed,
        "q_mixed": q_mixed,
        "i_demodulated": i_demodulated,
        "q_demodulated": q_demodulated,
        "combined_demodulated": demodulated_signal,
        "normalized": normalized_signal,
        "thresholded": thresholded_signal,
    }


def visualize_coherent_demodulation(wav_signal):
    """
    Visualize each step of the coherent demodulation process
    """
    # Perform demodulation and get intermediate signals
    _, signals = coherent_demodulate(wav_signal)

    # Create time array for plotting
    duration = len(wav_signal) / SAMPLE_RATE
    time_array = np.linspace(0, duration, len(wav_signal))

    # Original signal and local oscillator (just showing a small segment for visualization)
    plt.figure(figsize=(15, 12))

    # Plot original signal
    plt.subplot(5, 1, 1)
    plt.plot(time_array[:1000], wav_signal[:1000])
    plt.title("Original AM Signal (first 1000 samples)")
    plt.grid(True)

    # Plot I and Q mixed signals
    plt.subplot(5, 1, 2)
    plt.plot(time_array[:1000], signals["i_mixed"][:1000], label="I Mixed")
    plt.plot(time_array[:1000], signals["q_mixed"][:1000], label="Q Mixed", alpha=0.7)
    plt.title("Signal Mixed with Local Oscillators (I and Q)")
    plt.legend()
    plt.grid(True)

    # Plot filtered I and Q
    plt.subplot(5, 1, 3)
    plt.plot(time_array, signals["i_demodulated"], label="I Filtered")
    plt.plot(time_array, signals["q_demodulated"], label="Q Filtered", alpha=0.7)
    plt.title("Low-pass Filtered I and Q Signals")
    plt.legend()
    plt.grid(True)

    # Plot combined demodulated signal
    plt.subplot(5, 1, 4)
    plt.plot(time_array, signals["combined_demodulated"])
    plt.title("Combined Demodulated Signal")
    plt.grid(True)

    # Plot normalized and thresholded signal
    plt.subplot(5, 1, 5)
    plt.plot(time_array, signals["normalized"], label="Normalized")
    plt.plot(time_array, signals["thresholded"], label="Thresholded", alpha=0.7)
    plt.title("Normalized and Thresholded Signal")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Zoomed view for bit analysis
    section_length = min(5000, len(signals["thresholded"]))
    start_idx = find_start_index(signals["thresholded"]) - 500
    start_idx = max(0, min(start_idx, len(signals["thresholded"]) - section_length))

    plt.figure(figsize=(15, 8))
    plt.subplot(3, 1, 1)
    plt.plot(
        time_array[start_idx : start_idx + section_length],
        signals["combined_demodulated"][start_idx : start_idx + section_length],
    )
    plt.title("Zoomed Demodulated Signal")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(
        time_array[start_idx : start_idx + section_length],
        signals["normalized"][start_idx : start_idx + section_length],
    )
    plt.title("Zoomed Normalized Signal")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(
        time_array[start_idx : start_idx + section_length],
        signals["thresholded"][start_idx : start_idx + section_length],
    )
    plt.title("Zoomed Thresholded Signal")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Read the signal
    wav_signal = get_wav_signal(PATH_TO_WAV_FILE)

    # Perform coherent demodulation and visualize steps
    decoded_message, _ = coherent_demodulate(wav_signal)
    visualize_coherent_demodulation(wav_signal)

    print(f"Decoded message: {decoded_message}")

    # You might want to test different filter cutoff frequencies
    # by modifying CUT_OFF_FREQ in config_values.py

    # If synchronization is an issue, you might want to test different
    # carrier frequencies around your expected CARRIER_FREQ value
    print(f"Using carrier frequency: {CARRIER_FREQ} Hz")
    print(f"Using cutoff frequency: {CUT_OFF_FREQ} Hz")
    print(f"Using samples per symbol: {SAMPLES_PER_SYMBOL}")
