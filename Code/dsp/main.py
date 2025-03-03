from Code.dsp.transmitter_old import encode_and_modulate
from receiver import demodulate_and_decode, read_and_convert_wav_file, plot

# from image_decoding import convert_bits_to_image, show_picture
from config_values import SAMPLE_RATE, MESSAGE

if __name__ == "__main__":
    print("Encoding message...")
    modulated, time_array = encode_and_modulate(MESSAGE)

    print("Reading wav file...")
    freq_sample, signal_from_wave_file = read_and_convert_wav_file()

    print("Demodulating message...")
    decoded_message, envelope, bits, energy = demodulate_and_decode(
        signal_from_wave_file
    )

    print(f"Original message: {MESSAGE}")

    # prefix_from_message = decoded_message[0]
    # if (prefix_from_message == "p"):
    #     pixels = convert_bits_to_image(decoded_message[1:])
    #     show_picture(pixels)
    #     pass

    print(f"Decoded message: {decoded_message}")

    plot(signal_from_wave_file, envelope, energy, freq_sample)
