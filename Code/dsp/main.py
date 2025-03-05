from transmitterClass import Transmitter
from receiverClass import NonCoherentReceiver, CoherentReceiver
from config_values import PATH_TO_WAV_FILE, all_letters, SAMPLE_RATE
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Modified main function


def plot_demodulation_steps(receiver, debug_info, receiver_type):
    """Plots demodulation steps for both receiver types with improved layout."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 24))

    # Adjust subplot spacing
    plt.subplots_adjust(hspace=0.8, wspace=0.4)

    # Common first plot showing original signal in time and frequency domain
    plt.subplot(8, 2, 1)
    plt.plot(receiver.wav_signal, color="navy")
    plt.title("1. Original Input Signal (Time Domain)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Frequency domain of original signal
    plt.subplot(8, 2, 2)
    fft_original = np.fft.fft(receiver.wav_signal)
    freq = np.fft.fftfreq(len(receiver.wav_signal), d=1 / SAMPLE_RATE)
    plt.plot(
        freq[: len(freq) // 2], np.abs(fft_original[: len(freq) // 2]), color="navy"
    )
    plt.title("1. Original Input Signal (Frequency Domain)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    # Receiver-specific plots
    if receiver_type == "Non-Coherent":
        plots = [
            (
                debug_info["analytic"].real,
                "2. Analytic Signal (Real)",
                "darkgreen",
                False,
            ),
            (
                debug_info["analytic"].imag,
                "3. Analytic Signal (Imaginary)",
                "maroon",
                False,
            ),
            (debug_info["envelope"], "4. Envelope Detection", "darkorange", False),
            (debug_info["filtered"], "5. Low-Pass Filtered", "purple", True),
            (debug_info["normalized"], "6. Normalized & Cleaned", "teal", False),
            (debug_info["thresholded"], "7. Thresholded Output", "crimson", False),
        ]
    elif receiver_type == "Coherent":
        plots = [
            (
                debug_info["shifted"].real,
                "2. Frequency-Shifted (Real)",
                "darkgreen",
                True,
            ),
            (
                debug_info["shifted_imag"],
                "3. Negative Imaginary Component",
                "maroon",
                True,
            ),
            (debug_info["filtered"], "4. Low-Pass Filtered", "purple", True),
            (debug_info["normalized"], "5. Normalized Signal", "teal", False),
            (debug_info["thresholded"], "6. Thresholded Output", "crimson", False),
            (None, "7. Placeholder", "black", False),  # Empty plot for layout
        ]

    for i, (data, title, color, show_fft) in enumerate(plots, start=2):
        # Time domain plot
        plt.subplot(8, 2, (i - 1) * 2 + 1)
        if data is not None:
            plt.plot(data, color=color)
        plt.title(f"{title} (Time Domain)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(True)

        # Frequency domain plot if requested
        plt.subplot(8, 2, (i - 1) * 2 + 2)
        if data is not None and show_fft:
            fft_data = np.fft.fft(data)
            freq = np.fft.fftfreq(len(data), d=1 / SAMPLE_RATE)
            plt.plot(
                freq[: len(freq) // 2], np.abs(fft_data[: len(freq) // 2]), color=color
            )
            plt.title(f"{title} (Frequency Domain)")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.grid(True)
        else:
            plt.axis("off")  # Hide empty plots

    plt.tight_layout()
    plt.suptitle(f"{receiver_type} Demodulation Process", y=1.02, fontsize=16)
    plt.show()

def main():
    message = "AA"
    transmitter = Transmitter(message)
    transmitter.transmit()

    time.sleep(1)

    # Non-Coherent demodulation
    receiver_non_coherent = NonCoherentReceiver.from_wav_file(PATH_TO_WAV_FILE)
    message_nc, debug_nc = receiver_non_coherent.decode()
    print(f"Non-Coherent Decoded: {message_nc}")
    plot_demodulation_steps(receiver_non_coherent, debug_nc, "Non-Coherent")

    # Coherent demodulation
    receiver_coherent = CoherentReceiver.from_wav_file(PATH_TO_WAV_FILE)
    message_c, debug_c = receiver_coherent.decode()
    print(f"Coherent Decoded: {message_c}")
    plot_demodulation_steps(receiver_coherent, debug_c, "Coherent")

if __name__ == "__main__":
    main()
