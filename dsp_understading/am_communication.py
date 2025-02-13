import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt

SAMPLE_RATE = 1000000
CARRIER_FREQ = 100000
BIT_RATE = 10000

def encode_and_modulate(message, sample_rate=SAMPLE_RATE, carrier_freq=CARRIER_FREQ, bit_rate=BIT_RATE):
    """Encode text message and create AM modulated signal"""
    # Convert text to binary
    binary_message = ''
    for c in message:
        byte = format(ord(c), '08b')
        binary_message += byte
    
    print(f"Binary message: {binary_message}")
    
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
    shaped = 0.2 + 0.8 * (shaped - np.min(shaped)) / (np.max(shaped) - np.min(shaped))
    
    # Generate time array
    t = np.arange(len(shaped)) / sample_rate
    

    # After (direct frequency usage, ensure carrier_freq < sample_rate/2):
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    # Modulate
    modulated = shaped * carrier
    
    # Scale and convert to int16
    modulated = np.clip(modulated, -1, 1)
    modulated_int16 = (modulated * 32767).astype(np.int16)
    
    return modulated_int16, sample_rate, t, shaped

def demodulate_and_decode(modulated_signal, sample_rate=SAMPLE_RATE, carrier_freq=CARRIER_FREQ, bit_rate=BIT_RATE):
    """Demodulate AM signal and decode message"""
    # Convert to float
    modulated = modulated_signal.astype(np.float32) / 32767.0
    
    # Square law detector for envelope
    envelope = np.abs(signal.hilbert(modulated))
    
    # Low-pass filter design for envelope detection
    nyq = sample_rate / 2
    cutoff = bit_rate
    b, a = signal.butter(4, cutoff/nyq, btype='low')
    
    # Apply low-pass filter
    filtered = signal.filtfilt(b, a, envelope)
    
    # Normalize with offset removal
    filtered = filtered - np.mean(filtered)
    if np.max(filtered) - np.min(filtered) > 1e-10:  # Check signal variance
        normalized = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered))
    else:
        print("Error: Signal has insufficient variance")
        return "", filtered, []
    
    # Calculate samples per bit
    samples_per_bit = int(sample_rate / bit_rate)
    
    # Find start of data by looking for first significant transition
    energy = np.convolve(normalized, np.ones(samples_per_bit), 'valid')
    start_idx = np.where(energy > 0.5*np.max(energy))[0][0]
    
    # Extract bits using averaging
    bits = []
    for i in range(start_idx, len(normalized)-samples_per_bit, samples_per_bit):
        bit_sample = normalized[i:i+samples_per_bit]
        bit_value = 1 if np.mean(bit_sample) > 0.5 else 0
        bits.append(bit_value)
    
    # Convert bits to ASCII (8 bits per character)
    message = ""
    for i in range(0, len(bits)-7, 8):
        char_bits = bits[i:i+8]
        if len(char_bits) == 8:
            char_code = int(''.join(map(str, char_bits)), 2)
            if 32 <= char_code <= 126:  # Printable ASCII
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
    
    # Plot histogram of envelope values
    plt.subplot(4, 1, 3)
    plt.hist(envelope, bins=50)
    plt.title("Histogram of Envelope Values")
    plt.grid(True)
    
    # Plot bits if available
    if len(bits) > 0:
        plt.subplot(4, 1, 4)
        plt.step(range(len(bits)), bits, where='post')
        plt.title(f"Decoded Bits (total: {len(bits)})")
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parameters
    MESSAGE = "MY NAME IS MATHIAS"
    SAMPLE_RATE = SAMPLE_RATE  # 1 MHz
    CARRIER_FREQ = CARRIER_FREQ  # 100 MHz
    BIT_RATE = BIT_RATE  # 1 kbps
    
    print("Encoding message...")
    modulated, sample_rate, t, shaped = encode_and_modulate(
        MESSAGE, SAMPLE_RATE, CARRIER_FREQ, BIT_RATE
    )
    
    print(f"Signal stats - Max: {np.max(modulated)}, Min: {np.min(modulated)}")
    print(f"Modulation depth: {np.ptp(shaped):.2f}")
    
    print("Saving to WAV file...")
    wav.write("hello_world_am.wav", sample_rate, modulated)
    
    print("Demodulating message...")
    decoded_message, envelope, bits = demodulate_and_decode(
        modulated, sample_rate, CARRIER_FREQ, BIT_RATE
    )
    
    print(f"Original message: {MESSAGE}")
    print(f"Decoded message: {decoded_message}")
    
    # Plot first 10ms of signal
    samples_to_plot = int(0.01 * sample_rate)  # 10ms
    plot_debug(t, modulated, envelope, bits, samples_to_plot)