import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def compute_snr(signal_file, noise_file, bandpass_filter=None, plot_signals=False):
    """
    Compute the SNR from two recordings: one with signal+noise, one with noise only.

    Args:
        signal_file (str): Path to the .wav file with signal+noise.
        noise_file (str): Path to the .wav file with noise only.
        bandpass_filter (tuple, optional): (low_cutoff, high_cutoff) frequencies in Hz. If None, no filtering.
        plot_signals (bool): If True, plots both signals.

    Returns:
        float: Estimated SNR in dB.
    """
    from scipy.signal import butter, filtfilt

    # Load files
    sample_rate_signal, signal_plus_noise = wavfile.read(signal_file)
    sample_rate_noise, noise_only = wavfile.read(noise_file)

    # Check sample rates
    assert sample_rate_signal == sample_rate_noise, "Sample rates do not match."

    # If stereo, take only one channel
    if signal_plus_noise.ndim > 1:
        signal_plus_noise = signal_plus_noise[:, 0]
    if noise_only.ndim > 1:
        noise_only = noise_only[:, 0]

    # Optional: Apply bandpass filter
    if bandpass_filter is not None:
        low, high = bandpass_filter
        nyq = 0.5 * sample_rate_signal
        b, a = butter(4, [low/nyq, high/nyq], btype='band')
        signal_plus_noise = filtfilt(b, a, signal_plus_noise)
        noise_only = filtfilt(b, a, noise_only)

    # Compute power
    noise_power = np.mean(noise_only**2)
    total_power = np.mean(signal_plus_noise**2)

    # Compute signal power
    signal_power = total_power - noise_power

    if signal_power <= 0:
        raise ValueError("Signal power is non-positive. Something went wrong, maybe noise too strong?")

    # Compute SNR
    snr_db = 10 * np.log10(signal_power / noise_power)

    if plot_signals:
        time_signal = np.linspace(0, len(signal_plus_noise) / sample_rate_signal, len(signal_plus_noise))
        time_noise = np.linspace(0, len(noise_only) / sample_rate_noise, len(noise_only))

        # Convert amplitudes to dB
        # Add small constant to avoid log(0)
        eps = 1e-10
        signal_db = 20 * np.log10(np.abs(signal_plus_noise) + eps)
        noise_db = 20 * np.log10(np.abs(noise_only) + eps)

        plt.style.use('bmh')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Signal Analysis (SNR: {snr_db:.2f} dB)', fontsize=14, x=0.525, y=0.98)
        
        plt.subplots_adjust(top=0.90)

        # Signal + Noise plot in dB
        ax1.plot(time_signal, signal_db, 'b-', label='Signal + Noise', linewidth=1)
        ax1.set_title('Signal + Noise Recording', fontsize=12)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Amplitude [dB]')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        # Noise only plot in dB
        ax2.plot(time_noise, noise_db, 'r-', label='Noise Only', linewidth=1)
        ax2.set_title('Noise Only Recording', fontsize=12)
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Amplitude [dB]')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()

        # Add filter information if used
        if bandpass_filter is not None:
            fig.text(0.02, 0.02, f'Bandpass Filter: {bandpass_filter[0]}-{bandpass_filter[1]} Hz', 
                    fontsize=10, style='italic')

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()


    return snr_db

# Example usage
if __name__ == "__main__":
    # signal_path = "Code/dsp/data/signal_plus_noice_plastic_tank.wav"
    # signal_path = "Code/dsp/data/raw_data/71bb5db0-9772-4fec-b1f8-e9b7436c4f49.wav" # 1 vpp
    signal_path = "Code/dsp/data/raw_data/d7a33921-d295-48a1-87de-c3945c92f2ea.wav" # 0.5 vpp
    signal_path = "Code/dsp/data/raw_data/7bbb25bf-967e-436c-bd53-e9f55c148197.wav" # 0.25 vpp
    noise_path = "Code/dsp/data/noise_only_plastic_tank.wav"
    
    snr = compute_snr(signal_path, noise_path, plot_signals=True)
    print(f"Estimated SNR: {snr:.2f} dB")