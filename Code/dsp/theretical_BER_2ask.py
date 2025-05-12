import numpy as np
from scipy.special import erfc


# Simplified: source-level radius r = link distance d
# Absorption coefficient using Thorp's formula (f in kHz)
def alpha_thorp(f_khz):
    return (
        0.11 * f_khz**2 / (1 + f_khz**2)
        + 44 * f_khz**2 / (4100 + f_khz**2)
        + 2.75e-4 * f_khz**2
        + 0.003
    )


# Compute SNR (dB) with r = d and bandwidth = 2*Rb (carrier ± bit-rate)
def compute_snr_db(P_w, d_m, f_khz, Rb):
    # 1) Source level Slevel (dB), using r = d
    I = P_w / (4 * np.pi * d_m**2)
    Slevel = 10 * (np.log10(I) - np.log10(0.67e-18))
    # 2) Transmission loss Tloss (dB)
    Tloss = 20 * np.log10(d_m) + alpha_thorp(f_khz) * d_m * 1e-3
    # 3) Noise spectral density Nlevel (dB/Hz)
    Nlevel = 50 - 18 * np.log10(f_khz)
    # 4) Noise power over band B = 2 * Rb (dB)
    B = Rb
    Npow = Nlevel + 10 * np.log10(B)
    # 5) Directivity index (omnidirectional = 0)
    Dindex = 0
    # Final SNR in dB
    return Slevel - Tloss - Npow + Dindex


# Q-function via erfc
def Q(x):
    return 0.5 * erfc(x / np.sqrt(2))


# BER for 2-ASK (binary ASK)
def ber_2ask(snr_db):
    gamma_lin = 10 ** (snr_db / 10)
    return Q(np.sqrt(gamma_lin))


# Packet success probability
def packet_success_rate(P_w, d_m, f_khz, Rb, packet_size_bits):
    snr_db = compute_snr_db(P_w, d_m, f_khz, Rb)
    Pb = ber_2ask(snr_db)
    return (1 - Pb) ** packet_size_bits


# Example with your parameters
P = 0.007531573898450346  # transmitter power in W
d = 300.0  # link distance = source-level radius in m
fc = 2.0  # carrier frequency in kHz
Rb = 500  # bit rate in bits/s
m = 512  # packet size in bits

snr_db = compute_snr_db(P, d, fc, Rb)
ber = ber_2ask(snr_db)
psr = packet_success_rate(P, d, fc, Rb, m)

print(f"SNR (dB): {snr_db:.2f}")
print(f"BER: {ber:.2e}")
print(f"Packet Success Rate: {psr:.2e}")
