import matplotlib.pyplot as plt
import numpy as np

def plot_TRANSMITTER(square_wave, carrier_wave, modulated_wave, time_array):
    plt.figure(figsize=(12, 6))

    # Plot the square wave
    plt.subplot(3, 1, 1)
    plt.plot(time_array, square_wave, label='Square Wave')
    plt.title('Square Wave')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # Plot the carrier wave
    plt.subplot(3, 1, 2)
    plt.plot(time_array, carrier_wave, label='Carrier Wave', color='orange')
    plt.title('Carrier Wave')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # Plot the carrier wave
    plt.subplot(3, 1, 3)
    plt.plot(time_array, modulated_wave, label='Modulated Wave', color='pink')
    plt.title('Modulated Wave')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()



def visualize_transformations(signals, sample_rate, titles):
    time_array = np.linspace(0, len(signals[0]) / sample_rate, len(signals[0]))

    plt.figure(figsize=(12, 8))

    for i, (signal, title) in enumerate(zip(signals, titles), 1):
        plt.subplot(len(signals), 1, i)
        plt.plot(time_array, signal)
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid()

    plt.tight_layout()
    plt.show()
