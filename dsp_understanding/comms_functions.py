import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft, ifft


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


# def fft(signal):
#     """
#     Perform Fast Fourier Transform on the signal.
#     :param signal: The signal is an array to perform FFT on.
#     :return: the signal converted to frequency domain. With magnitude as y-axis and frequency as x-axis.
#
#     # NOTE: This is a recursive implementation and a key aspect is that length of the signal is a power of 2. For the future we shall use signals with lengths that are not of power of 2 and we shall instead make use of the numpy implementation of the FFT.
#     """
#     n = len(signal)  # n is a power of 2
#     if n == 1:
#         return signal
#     omega = np.exp(2 * np.pi * 1j / n)
#
#     signal_even, signal_odd = signal[::2], signal[1::2]
#     y_even, y_odd = fft(signal_even), fft(signal_odd)
#     y = np.zeros(n, dtype=complex)
#     for k in range(n // 2):
#         y[k] = y_even[k] + omega**k * y_odd[k]
#         y[k + n // 2] = y_even[k] - omega**k * y_odd[k]
#     return y
#
#
# def ifft(fft_signal):
#     """
#     Perform Inverse Fast Fourier Transform on the signal.
#     :param fft_signal: The signal is an array to perform IFFT on.
#     :return: the signal converted to time domain. With i.e. voltage as y-axis and time as x-axis.
#     """
#     n = len(fft_signal)  # n is a power of 2
#     if n == 1:
#         return fft_signal
#     omega = (1 / n) * np.exp(-2 * np.pi * 1j / n)
#     fft_signal_even, fft_signal_odd = fft_signal[::2], fft_signal[1::2]
#     y_even, y_odd = ifft(fft_signal_even), ifft(fft_signal_odd)
#     y = np.zeros(n, dtype=complex)
#     for k in range(n // 2):
#         y[k] = y_even[k] + omega**k * y_odd[k]
#         y[k + n // 2] = y_even[k] - omega**k * y_odd[k]
#     return y
#

### Now we will construct the Hilbert transform of a signal
### using the FFT and IFFT functions defined above.


def hilbert_transform_scratch(signal):
    """
    Perform Hilbert Transform on the signal. A Hilbert transform in mathematics is a type of linear operator that creates an analytic signal from a real-valued signal. The procedure is also known as analytic continuation. We will make use of it to get the envelope of the signal. What is an envelope of a signal you ask? It is a smooth curve outlining the extremes of the signal. see https://en.wikipedia.org/wiki/Envelope_(waves) for more details.

    :param signal: The signal is an array to perform Hilbert Transform on.
    :return: the hilbert transform of the signal.

    Variables created:
    - n: length of the signal
    - m: number of elements to zero out.
    - fft_signal: Fast Fourier Transform of the signal
    - ifft_signal: Inverse Fast Fourier Transform of the signal
    """
    n = len(signal)
    # Perform FFT on the signal
    fft_signal = fft(signal)
    m = n - n // 2 - 1
    # Zero out the negative frequencies
    fft_signal[n // 2 + 1 :] = [0] * m
    # Double the fft energy except at the zero frequency
    fft_signal[1 : n // 2] = 2 * fft_signal[1 : n // 2]
    # Perform IFFT on the signal
    ifft_signal = ifft(fft_signal)
    return ifft_signal


def amplitude_envelope(signal, frame_size, hop_length):
    """
    Compute the amplitude envelope of a signal. (This is a simplified version of the achieving an envelope from the Hilbert transform)
    :param signal: The signal is an array to perform amplitude envelope on.
    :param frame_size: The size of the frame.
    :param hop_size: The size of the hop.
    :return: the amplitude envelope of the signal.
    """
    amplitude_envelope = []
    for i in range(0, len(signal), hop_length):
        current_frame_amplitude_envelope = max(signal[i : i + frame_size])
        amplitude_envelope.append(current_frame_amplitude_envelope)
    return np.array(amplitude_envelope)


def get_envelope(signal):
    """
    Get the envelope of the signal.
    :param signal: The signal is an array to perform Hilbert Transform on.
    :return: the envelope of the signal.
    """
    return np.abs(hilbert_transform_scratch(signal))


if __name__ == "__main__":
    # Testing the Hilbert from scratch function
    # Create a signal with length of power of 2.
    x = np.random.normal(-1, 1, size=44100)
    hilbert_transform = hilbert_transform_scratch(x)
    # compare the result with the scipy implementation
    hilbert_transform_scipy = signal.hilbert(x)
    print(np.allclose(hilbert_transform, hilbert_transform_scipy))
    print("done testing hilbert transform")
    print(
        f"Comparison of the 2 hilbert transforms: {hilbert_transform} and {hilbert_transform_scipy}"
    )
    # Testing the envelope function
    envelope = get_envelope(x)
    envelope_scipy = np.abs(hilbert_transform_scipy)
    ae = amplitude_envelope(x, 1024, 512)
    plt.figure(figsize=(10, 6))

    plt.subplot(4, 1, 1)
    plt.plot(x, label="Original signal", alpha=0.5)
    plt.subplot(4, 1, 2)
    plt.plot(envelope, label="Envelope from scratch", alpha=0.5)
    plt.subplot(4, 1, 3)
    plt.plot(envelope_scipy, label="Envelope from scipy", alpha=0.5)
    plt.subplot(4, 1, 4)
    plt.plot(ae, label="Amplitude envelope", alpha=0.5)
    plt.legend()
    plt.show()
    print("done testing envelope")
