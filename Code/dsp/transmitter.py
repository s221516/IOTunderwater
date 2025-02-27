import numpy as np
import scipy.io.wavfile as wav
from config_values import BIT_RATE, SAMPLE_RATE, CARRIER_FREQ, NOISE_AMPLITUDE, PATH_TO_WAV_FILE, SAMPLE_RATE_FOR_WAV_FILE

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


