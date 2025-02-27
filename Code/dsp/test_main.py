from receiver import demodulate_and_decode, read_and_convert_wavefile, convert_to_mono, plot
# from image_decoding import convert_bits_to_image, show_picture
from record_audio import create_wav_file_from_recording


if __name__ == "__main__":
    create_wav_file_from_recording()

    print("Reading wav file...")
    freq_sample, signal_from_wave_file = read_and_convert_wavefile()
    signal_from_wave_file_mono = convert_to_mono(signal_from_wave_file)

    print("Decode signal...")
    decoded_message, envelope, bits, energy = demodulate_and_decode(signal_from_wave_file_mono)

    print(f"Decoded message: {decoded_message}")
    
    plot(signal_from_wave_file_mono, envelope, energy, freq_sample)
