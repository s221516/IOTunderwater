import matplotlib.pyplot as plt
import config
import librosa
import numpy as np
import scipy.io.wavfile as wavfile
import pandas as pd



#make plotting class
df = pd.read_csv("Code/dsp/data/csv_logs/Logging_Data.csv")
class SignalPlotter:
    def __init__(self, id):
        self.id = id
        _ , self.wav_signal = wavfile.read("Code/dsp/data/raw_data/" + id + ".wav")

    def plot_spectrogram(self, ax=None):
        """Plots the spectrogram of the received WAV signal."""
        if ax is None:
            ax = plt.gca()
            
        if self.wav_signal is None:
            print("No signal to plot spectrogram for.")
            return

        # Define hop_length (adjust as needed for desired time resolution)
        hop_length = 256 
        # Calculate the STFT (Short-Time Fourier Transform)
        stft_result = librosa.stft(self.wav_signal.astype(float), hop_length=hop_length)
        # Get the magnitude of the STFT result
        # without decibels
        amplitude_spectrogram = np.abs(stft_result)
        # Convert to decibels (optional, for better visualization)
        amplitude_spectrogram_db = librosa.amplitude_to_db(amplitude_spectrogram, ref=np.max)

        # Display the spectrogram using decibels
        librosa.display.specshow(amplitude_spectrogram_db, sr=config.SAMPLE_RATE, hop_length=hop_length, x_axis="time", y_axis="hz") 
        plt.colorbar(label="Amplitude (dB)") # Updated colorbar label

        ax.set_title(f"Spectrogram (Decibels) for signal with carrier {df[df['id'] == id]['carrier_frequency']}Hz and bitrate {df[df['id'] == id]['bitrate']}Hz")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, 20000) # Limit y-axis to 20kHz
        
        plt.show()
            
    def plot_wave_in_frequency_domain(self, ax=None, color="b", alpha=0.5, label=None):
        """
        Plots the frequency domain representation of the received WAV signal.
        @param wave: The received WAV signal.
        @param ax: The axis to plot on. If None, uses the current axis.
        @param color: The color of the plot line.
        @param alpha: The transparency of the plot line.
        @param label: The label for the plot line for the legend.
        @return: None
        """
        if ax is None:
            ax = plt.gca()

        wave_f = np.fft.fft(self.wav_signal)
        wave_f = np.fft.fftshift(wave_f)
        frequencies_x_axis = np.arange(
            config.SAMPLE_RATE / -2, config.SAMPLE_RATE / 2, config.SAMPLE_RATE / len(self.wav_signal)
        )

        frequency_magnitudes = np.abs(wave_f)
        # Make it to decibels
        frequency_magnitudes = 10 * np.log10(frequency_magnitudes / np.max(frequency_magnitudes))

        # only plot the positive frequencies
        positive_frequencies = frequencies_x_axis > 0
        frequencies_x_axis = frequencies_x_axis[positive_frequencies]
        frequency_magnitudes = frequency_magnitudes[positive_frequencies]

        ax.plot(frequencies_x_axis, frequency_magnitudes, "-", color=color, alpha=alpha, label=label) 
        
        plt.show()

    def plot_wave_in_time_domain(self, ax=None, color="orange", alpha=0.5):

        """
        Plots the time domain representation of the received WAV signal.
        @param wave: The received WAV signal.
        @param l: The label for the plot line for the legend.
        @param ax: The axis to plot on. If None, uses the current axis.
        @param color: The color of the plot line.
        @return: None
        """
        if ax is None:
            ax = plt.gca()

        time_array = np.arange(len(self.wav_signal)) / config.SAMPLE_RATE
        ax.plot(time_array, self.wav_signal, color=color, label="Wave plot", alpha=alpha)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Time Domain Signal")
        ax.grid(True)

        plt.show()



if __name__ == "__main__":
    id = "3d3ff86e-3037-4f4e-84dc-0312c54b5f01"
    plotter = SignalPlotter(id)
    plotter.plot_spectrogram()
    plotter.plot_wave_in_frequency_domain()
    plotter.plot_wave_in_time_domain()