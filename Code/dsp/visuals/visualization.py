# File: visualization.py
# Optimized visualization with better layout and sizing

import matplotlib.pyplot as plt
import numpy as np
import config
from visuals.plotting import plot_wave_in_frequency_domain, plot_wave_in_time_domain


def get_bits_from_thresholded(thresholded_signal):
    """
    Extract bits from thresholded signal - same logic as in Receiver.get_bits
    but extracted to avoid circular imports
    """
    bits = []
    for i in range(0, len(thresholded_signal), config.SAMPLES_PER_SYMBOL):
        if i + config.SAMPLES_PER_SYMBOL <= len(thresholded_signal):
            mu = np.mean(thresholded_signal[i : i + config.SAMPLES_PER_SYMBOL])
            bits.append(1 if mu > 0.5 else 0)
    return bits


def create_processing_visualization(receiver, message, debug_info):
    """
    Create a comprehensive visualization of signal processing steps with shared axes
    and optimized layout for standard window sizes
    """
    # Smaller overall figure size that works better on standard displays
    fig = plt.figure(figsize=(10, 6))

    # Adjust the subplot parameters for tighter spacing
    plt.subplots_adjust(
        left=0.08, right=0.95, bottom=0.08, top=0.92, wspace=0.25, hspace=0.4
    )

    # Create a more compact grid - 3 rows x 3 columns instead of 4x2
    # This creates smaller subplots that fit better

    # Time domain plots in first column
    ax1 = plt.subplot(3, 3, 1)  # Received signal
    ax4 = plt.subplot(3, 3, 4)  # Normalized and thresholded
    ax7 = plt.subplot(3, 3, 7)  # Extracted bits
    time_domain_axes = [ax1, ax4, ax7]

    # Frequency domain plots in second column
    ax2 = plt.subplot(3, 3, 2)  # Received signal in freq domain
    ax5 = plt.subplot(
        3, 3, 5, sharex=ax2, sharey=ax2
    )  # Intermediate signal in freq domain
    ax8 = plt.subplot(3, 3, 8, sharex=ax2, sharey=ax2)  # Filtered signal in freq domain
    freq_domain_axes = [ax2, ax5, ax8]

    # Results and extra info in third column
    ax3 = plt.subplot(3, 3, 3)  # Can be used for signal details
    ax6 = plt.subplot(3, 3, 6)  # Message display
    ax9 = plt.subplot(3, 3, 9)  # Binary display

    # Add a title for the entire figure
    plt.suptitle("Signal Processing Steps and Analysis", fontsize=14, fontweight="bold")

    # Check receiver type by looking at debug_info keys
    is_non_coherent = "envelope" in debug_info

    plot_wave_in_time_domain(receiver.wav_signal, "Received", ax=ax1, color="g")

    if is_non_coherent:
        # NonCoherentReceiver plots
        plot_wave_in_time_domain(debug_info["envelope"], "Envelope", ax=ax1, color="b")
        plot_wave_in_time_domain(debug_info["filtered"], "Filtered", ax=ax1, color="r")

        plot_wave_in_time_domain(
            debug_info["normalized"], "Normalized", ax=ax4, color="g"
        )
        plot_wave_in_time_domain(
            debug_info["thresholded"], "Thresholded", ax=ax4, color="r"
        )

        # Additional info in ax3
        ax3.plot([], [], color="b", label="Envelope")
        ax3.plot([], [], color="r", label="Filtered")
        ax3.plot([], [], color="g", label="Normalized")
        ax3.legend(loc="center")
        ax3.set_title("Signal Legend")
        ax3.axis("off")
    else:
        # CoherentReceiver plots
        plot_wave_in_time_domain(
            debug_info["shifted_imag"], "Shifted", ax=ax1, color="b"
        )
        plot_wave_in_time_domain(debug_info["filtered"], "Filtered", ax=ax1, color="r")

        plot_wave_in_time_domain(
            debug_info["normalized"], "Normalized", ax=ax4, color="g"
        )
        plot_wave_in_time_domain(
            debug_info["thresholded"], "Thresholded", ax=ax4, color="b"
        )

        # Additional info in ax3
        ax3.plot([], [], color="b", label="Shifted")
        ax3.plot([], [], color="r", label="Filtered")
        ax3.plot([], [], color="g", label="Normalized")
        ax3.legend(loc="center")
        ax3.set_title("Signal Legend")
        ax3.axis("off")

    # Plot bits in time domain - more compact format
    bits = get_bits_from_thresholded(debug_info["thresholded"])
    bit_times = np.arange(len(bits)) * config.SAMPLES_PER_SYMBOL / config.SAMPLE_RATE
    bit_values = np.array(bits)
    ax7.stem(bit_times, bit_values, linefmt="g-", markerfmt="go", basefmt=" ")

    # Plot frequency domain signals
    plot_wave_in_frequency_domain(receiver.wav_signal, ax=ax2, color="g")

    if is_non_coherent:
        plot_wave_in_frequency_domain(debug_info["envelope"], ax=ax5, color="b")
        plot_wave_in_frequency_domain(debug_info["filtered"], ax=ax8, color="r")
    else:
        plot_wave_in_frequency_domain(debug_info["shifted_imag"], ax=ax5, color="b")
        plot_wave_in_frequency_domain(debug_info["filtered"], ax=ax8, color="r")

    # Show the decoded message
    ax6.axis("off")
    ax6.text(
        0.5,
        0.5,
        f"Decoded message:\n{message}",
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8),
    )

    # Add a binary representation of the message
    bits_text = []
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            byte = bits[i : i + 8]
            if any(byte): # Checks if just one of the bits is a 1, thereby removing the byte blocks where they are all 0
                bits_text.append("".join(map(str, byte)))

    bits_formatted = " ".join(bits_text)
    ax9.axis("off")
    ax9.text(
        0.5,
        0.5,
        f"Binary representation:\n{bits_formatted}",
        fontsize=9,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="blue", alpha=0.8),
    )

    # Set titles with smaller font size to save space
    ax1.set_title("Received & Processed Signal", fontsize=10)
    ax2.set_title("Signal in Frequency Domain", fontsize=10)
    ax4.set_title("Normalized & Threshold", fontsize=10)
    ax5.set_title("Envelope", fontsize=10)
    ax7.set_title("Extracted Bits", fontsize=10)
    ax8.set_title("Filtered Frequency", fontsize=10)
    ax6.set_title("Decoded Message", fontsize=10)

    # Set common properties for time domain plots
    for ax in time_domain_axes:
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.grid(True, alpha=0.5)
        # Make legend smaller and place it in upper right to save space
        if ax != ax7:  # Don't need legend for bit stem plot
            ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(axis="both", which="major", labelsize=8)

    # Set common properties for frequency domain plots
    for ax in freq_domain_axes:
        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_ylabel("Magnitude", fontsize=9)
        ax.grid(True, alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=8)

    return fig

def create_frequency_domain_visualization(receiver, debug_info):
    """
    Create a visualization focusing on the frequency domain signals:
    original, envelope, and filtered.
    """
    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    # Adjust the subplot parameters for tighter spacing
    plt.subplots_adjust(
        left=0.06, right=0.98, bottom=0.15, top=0.85, wspace=0.2
    )

    # Add a title for the entire figure
    plt.suptitle("Frequency Domain Analysis", fontsize=14, fontweight="bold")

    # Plot original received signal in frequency domain
    plot_wave_in_frequency_domain(receiver.wav_signal, ax=axes[0], color="g")
    axes[0].set_title("Original Signal", fontsize=10)

    # Check if envelope data is available (for NonCoherentReceiver)
    if "envelope" in debug_info:
        plot_wave_in_frequency_domain(debug_info["envelope"], ax=axes[1], color="b")
        axes[1].set_title("Envelope Signal", fontsize=10)
    elif "shifted_imag" in debug_info: # Handle CoherentReceiver case if needed
         plot_wave_in_frequency_domain(debug_info["shifted_imag"], ax=axes[1], color="b")
         axes[1].set_title("Shifted Signal", fontsize=10)
    else:
        axes[1].set_title("Intermediate Signal (N/A)", fontsize=10)
        axes[1].text(0.5, 0.5, "No intermediate signal data", ha='center', va='center', transform=axes[1].transAxes)


    # Plot filtered signal in frequency domain
    if "filtered" in debug_info:
        plot_wave_in_frequency_domain(debug_info["filtered"], ax=axes[2], color="r")
        axes[2].set_title("Filtered Signal", fontsize=10)
    else:
        axes[2].set_title("Filtered Signal (N/A)", fontsize=10)
        axes[2].text(0.5, 0.5, "No filtered signal data", ha='center', va='center', transform=axes[2].transAxes)


    # Set common properties for frequency domain plots
    for ax in axes:
        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_ylabel("Magnitude", fontsize=9)
        ax.grid(True, alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=8)
        # Limit x-axis if needed, e.g., based on expected frequency range
        # ax.set_xlim([0, config.CARRIER_FREQ * 2]) # Example limit

    # Ensure the y-label is only shown on the first plot
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")


    return fig