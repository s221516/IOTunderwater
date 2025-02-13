import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read, write
from scipy.signal import butter, filtfilt, stft


def analyze_jackhammer_frequency(data, fs, nperseg=2048):
    """
    Analyze the frequency components to identify jackhammer peaks
    """
    # Compute STFT
    f, _, Zxx = stft(data, fs=fs, nperseg=nperseg)

    # Get magnitude spectrum
    magnitude = np.abs(Zxx)

    # Average over time to get frequency profile
    avg_spectrum = np.mean(magnitude, axis=1)

    return f, avg_spectrum


def create_notch_filter(fs, center_freq, q_factor=30):
    """
    Create a notch filter for specific frequencies
    """
    nyquist = fs / 2
    freq = center_freq / nyquist

    # Create notch filter coefficients
    b, a = butter(
        2,
        [freq - freq / q_factor, freq + freq / q_factor],
        btype="bandstop",
        analog=False,
    )
    return b, a


def remove_jackhammer_noise(audio_path, output_path):
    """
    Remove jackhammer noise using targeted frequency filtering
    """
    # Load audio
    fs, data = read(audio_path)

    # Convert to float32 for processing
    data_float = data.astype(np.float32)

    # If stereo, process each channel
    if len(data.shape) > 1:
        channels = []
        for channel in range(data.shape[1]):
            # Analyze frequency content
            freqs, spectrum = analyze_jackhammer_frequency(data_float[:, channel], fs)

            # Typical jackhammer frequencies are between 30-200 Hz
            mask = (freqs >= 0) & (freqs <= 500)
            jackhammer_freqs = freqs[mask]
            jackhammer_spectrum = spectrum[mask]

            # Find peaks in jackhammer frequency range
            from scipy.signal import find_peaks

            peaks, _ = find_peaks(
                jackhammer_spectrum, height=np.mean(jackhammer_spectrum)
            )

            # Apply notch filters for each major peak
            processed_channel = data_float[:, channel].copy()
            for peak_idx in peaks:
                peak_freq = jackhammer_freqs[peak_idx]
                b, a = create_notch_filter(fs, peak_freq)
                processed_channel = filtfilt(b, a, processed_channel)

            channels.append(processed_channel)

        # Combine channels
        processed_data = np.column_stack(channels)
    else:
        # Mono audio processing
        freqs, spectrum = analyze_jackhammer_frequency(data_float, fs)
        mask = (freqs >= 30) & (freqs <= 200)
        jackhammer_freqs = freqs[mask]
        jackhammer_spectrum = spectrum[mask]

        peaks, _ = find_peaks(jackhammer_spectrum, height=np.mean(jackhammer_spectrum))

        processed_data = data_float.copy()
        for peak_idx in peaks:
            peak_freq = jackhammer_freqs[peak_idx]
            b, a = create_notch_filter(fs, peak_freq)
            processed_data = filtfilt(b, a, processed_data)

    # Normalize and convert back to int16
    processed_data = np.int16(processed_data * 32767 / np.max(np.abs(processed_data)))

    # Save processed audio
    write(output_path, fs, processed_data)

    return processed_data, fs


def plot_before_after(original_data, processed_data, fs):
    """
    Plot spectrograms of original and processed audio
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Original spectrogram
    f, t, Zxx = stft(
        original_data[:, 0] if len(original_data.shape) > 1 else original_data,
        fs=fs,
        nperseg=2048,
    )
    ax1.pcolormesh(t, f, np.abs(Zxx), shading="gouraud")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_title("Original Audio Spectrogram")

    # Processed spectrogram
    f, t, Zxx = stft(
        processed_data[:, 0] if len(processed_data.shape) > 1 else processed_data,
        fs=fs,
        nperseg=2048,
    )
    ax2.pcolormesh(t, f, np.abs(Zxx), shading="gouraud")
    ax2.set_ylabel("Frequency [Hz]")
    ax2.set_xlabel("Time [sec]")
    ax2.set_title("Processed Audio Spectrogram")

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    input_file = "jackhammer.wav"
    output_file = "jackhammer_removed.wav"

    # Load original audio for comparison
    fs_orig, original_data = read(input_file)

    # Process audio
    processed_data, fs = remove_jackhammer_noise(input_file, output_file)

    # Plot spectrograms
    plot_before_after(original_data, processed_data, fs)
