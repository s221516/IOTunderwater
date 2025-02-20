import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt

SAMPLE_RATE = 200000 # 50 Khz
CARRIER_FREQ = 15200 #  15200 Hz
BIT_RATE = 1000 # 1 Khz
NOISE_AMPLITUDE = 0.000001

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def encode_and_modulate(message, sample_rate=SAMPLE_RATE, carrier_freq=CARRIER_FREQ, bit_rate=BIT_RATE, noise_amplitude = NOISE_AMPLITUDE):
    """Encode text message and create AM modulated signal"""
    # Convert text to binary
    binary_message = ''
    for c in message:
        byte = format(ord(c), '08b')
        binary_message += byte
    
    # print(f"Binary message: {binary_message}")
    
    # Calculate timing parameters
    duration_per_bit = 1 / bit_rate
    samples_per_bit = int(sample_rate * duration_per_bit)
    
    # Create baseband signal with oversampling
    bits = np.array([int(b) for b in binary_message])
    baseband = np.repeat(bits, samples_per_bit)
    
    # Add padding at start and end
    padding = np.zeros(samples_per_bit * 10)
    baseband = np.concatenate([padding, baseband, padding])
    
    # Apply pulse shaping (raised cosine)
    alpha = 0.35  # Roll-off factor
    symbol_length = samples_per_bit
    t = np.arange(-4*symbol_length, 4*symbol_length)
    h = np.sinc(t/symbol_length) * np.cos(np.pi*alpha*t/symbol_length) / (1 - (2*alpha*t/symbol_length)**2)
    h[np.isnan(h)] = 1  # Fix division by zero
    h = h / np.sum(h)  # Normalize
    
    # Apply pulse shaping
    shaped = np.convolve(baseband, h, 'same')
    
    # Scale to ensure good modulation depth (0.2 to 1.0)
    shaped = 0.2 + 0.85 * (shaped - np.min(shaped)) / (np.max(shaped) - np.min(shaped))
    
    # Generate time array
    t = np.arange(len(shaped)) / sample_rate

    # After (direct frequency usage, ensure carrier_freq < sample_rate/2):
    carrier = np.sin(2 * np.pi * carrier_freq * t)

    # Modulate
    modulated = shaped * carrier
    
    # Gives the first 15 elements of our amplitudes
    # print(modulated[:20])

    # Add Gaussion noise, find out if this is the correct interval, maybe its -x to x and not 0 to x
    noise = np.random.normal(0, NOISE_AMPLITUDE, len(modulated))
    modulated_with_noise = modulated + noise

    # Compute SNR
    snr = compute_snr(modulated, noise)
    # print(f"Signal-to-noise ratio: {snr} dB")
    B = bit_rate
    S = np.mean(modulated ** 2)
    N = np.mean(noise ** 2)
    shannon_limit = B*np.log2(1 + (S / N))
    # print(f"Shannon limit: {shannon_limit}")

    
    return modulated_with_noise, sample_rate, t, shaped

def demodulate_and_decode(modulated, sample_rate=SAMPLE_RATE, carrier_freq=CARRIER_FREQ, bit_rate=BIT_RATE):
    """Demodulate AM signal and decode message"""

    # TODO: understand math behind hilbert transform, keyword: analytic signal
    # make our own implementation of the hilbert transform
    envelope = np.abs(signal.hilbert(modulated))
    
    cutoff = bit_rate
    signal_post_filter = butter_lowpass_filter(envelope, cutoff, sample_rate, order = 4)

    signal_post_filter = signal_post_filter - np.mean(signal_post_filter)

    s_min = np.min(signal_post_filter)
    s_max = np.max(signal_post_filter)
    error = 1e-12

    normalized = (signal_post_filter - s_min) / (s_max - s_min + error)
    
    # Calculate samples per bit
    samples_per_bit = int(sample_rate / bit_rate)

    # Create a matched filter for bit detection
    matched_filter = np.ones(samples_per_bit)
    
    # Find start of data by looking for first significant transition
    energy = np.convolve(normalized, matched_filter, 'valid')

    # maybe add a noise filter here, but by slightly lowering the max energy required it got super clear
    start_of_valid_data_array = np.where(energy > 0.35*np.max(energy))[0]
    start_index = start_of_valid_data_array[0]

    # makes the bitstring for the valid data array
    bits = []
    for i in range(start_index, len(normalized) - samples_per_bit, samples_per_bit):
        bit_sample = normalized[i:i+samples_per_bit]
        bit_value = 1 if np.sum(bit_sample) > (samples_per_bit / 2) else 0
        bits.append(bit_value)
    

    # Convert bits to ASCII (8 bits per character)
    message = ""
    for i in range(0, len(bits)-7, 8):
        char_bits = bits[i:i+8]
        if len(char_bits) == 8:
            char_code = int(''.join(map(str, char_bits)), 2)
            if 32 <= char_code <= 126:  # Printable ASCII
                # print(f"index: {i / 8}, char_bits: {char_bits}")
                message += chr(char_code)
    
    return message, normalized, bits

def plot_debug(t, modulated, envelope, bits, samples_to_plot=None):
    """Create debug plots with signal information"""
    if samples_to_plot is None:
        samples_to_plot = len(t)
    
    plt.figure(figsize=(15, 12))
    
    # Plot modulated signal
    plt.subplot(4, 1, 1)
    plt.plot(t[:samples_to_plot], modulated[:samples_to_plot])
    plt.title(f"Modulated Signal (max={np.max(modulated):.2f}, min={np.min(modulated):.2f})")
    plt.grid(True)
    
    # Plot envelope
    plt.subplot(4, 1, 2)
    plt.plot(t[:samples_to_plot], envelope[:samples_to_plot])
    plt.title(f"Envelope (max={np.max(envelope):.2f}, min={np.min(envelope):.2f})")
    plt.grid(True)
    
    # # Plot histogram of envelope values
    # plt.subplot(4, 1, 3)
    # plt.hist(envelope, bins=50)
    # plt.title("Histogram of Envelope Values")
    # plt.grid(True)
    
    # Plot bits if available
    if len(bits) > 0:
        plt.subplot(4, 1, 3)
        plt.step(range(len(bits)), bits, where='post')
        plt.title(f"Decoded Bits (total: {len(bits)})")
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def compare_strings(original, decoded):
    """Compare two strings and print out where they differ"""
    differences = []
    min_length = min(len(original), len(decoded))
    
    for i in range(min_length):
        if original[i] != decoded[i]:
            differences.append((i, original[i], decoded[i]))
    
    # Check if one string is longer than the other
    if len(original) > len(decoded):
        for i in range(len(decoded), len(original)):
            differences.append((i, original[i], None))
    elif len(decoded) > len(original):
        for i in range(len(original), len(decoded)):
            differences.append((i, None, decoded[i]))
    
    return differences

def compute_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    return 10*np.log10((signal_power - noise_power) / noise_power)


if __name__ == "__main__":
    # Parameters
    plastic_bag_lyrics = """Do you ever feel like a plastic bag
    Drifting through the wind
    Wanting to start again?
    Do you ever feel, feel so paper-thin
    Like a house of cards, one blow from caving in?

    Do you ever feel already buried deep?
    Six feet under screams but no one seems to hear a thing
    Do you know that there's still a chance for you
    'Cause there's a spark in you?

    You just gotta ignite the light and let it shine
    Just own the night like the 4th of July

    'Cause, baby, you're a firework
    Come on, show 'em what you're worth
    Make 'em go, "Ah, ah, ah"
    As you shoot across the sky

    Baby, you're a firework
    Come on, let your colors burst
    Make 'em go, "Ah, ah, ah"
    You're gonna leave 'em all in awe, awe, awe

    You don't have to feel like a wasted space
    You're original, cannot be replaced
    If you only knew what the future holds
    After a hurricane comes a rainbow

    Maybe a reason why all the doors are closed
    So you could open one that leads you to the perfect road
    Like a lightning bolt your heart will glow
    And when it's time you'll know

    You just gotta ignite the light and let it shine
    Just own the night like the 4th of July

    'Cause, baby, you're a firework
    Come on, show 'em what you're worth
    Make 'em go, "Ah, ah, ah"
    As you shoot across the sky

    Baby, you're a firework
    Come on, let your colors burst
    Make 'em go, "Ah, ah, ah"
    You're gonna leave 'em all in awe, awe, awe

    Boom, boom, boom
    Even brighter than the moon, moon, moon
    It's always been inside of you, you, you
    And now it's time to let it through, -ough, -ough

    'Cause, baby, you're a firework
    Come on, show 'em what you're worth
    Make 'em go, "Ah, ah, ah"
    As you shoot across the sky

    Baby, you're a firework
    Come on, let your colors burst
    Make 'em go, "Ah, ah, ah"
    You're gonna leave 'em all in awe, awe, awe

    Boom, boom, boom
    Even brighter than the moon, moon, moon
    Boom, boom, boom
    Even brighter than the moon, moon, moon"""
    all_letters = "The quick brown fox jumps over the lazy dog while vexd zebras fight for joy! @#$%^&()_+[]{}|;:,.<>/?~` \ The 5 big oxen love quick daft zebras & dogs.>*"
    small_test = "mo"
    
    MESSAGE = small_test

    print("Encoding message...")
    modulated, sample_rate, t, shaped = encode_and_modulate(
        MESSAGE, SAMPLE_RATE, CARRIER_FREQ, BIT_RATE
    )
    
    # print(f"Signal stats - Max: {np.max(modulated)}, Min: {np.min(modulated)}")
    # print(f"Modulation depth: {np.ptp(shaped):.2f}")
    
    print("Saving to WAV file...")
    wav.write("hello_world_am.wav", sample_rate, modulated)
    
    print("Demodulating message...")
    decoded_message, envelope, bits = demodulate_and_decode(
        modulated, sample_rate, CARRIER_FREQ, BIT_RATE
    )
    
    print(f"Original message: {MESSAGE}")
    print(f"Decoded message: {decoded_message}")
    differences = compare_strings(MESSAGE, decoded_message)
    if differences:
        print("Differences found:")
        for index, orig_char, dec_char in differences:
            print(f"Position {index}: Original='{orig_char}', Decoded='{dec_char}'")
    else:
        print("No errors")
    
    # Plot first 10ms of signal
    samples_to_plot = int(0.1 * sample_rate)  # 10ms
    plot_debug(t, modulated, envelope, bits, samples_to_plot)


