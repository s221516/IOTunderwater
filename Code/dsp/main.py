from transmitter import encode_and_modulate
from receiver import demodulate_and_decode, read_wavefile, plot_debug
from image_decoding import convert_bits_to_image, show_picture
from config_values import SAMPLE_RATE, MESSAGE


if __name__ == "__main__":
    print("Encoding message...")
    modulated, time_array = encode_and_modulate(MESSAGE)

    print("Reading wav file...")
    freq_sample, signal_from_wave_file = read_wavefile()

    print("Demodulating message...")
    decoded_message, envelope, bits, energy = demodulate_and_decode(signal_from_wave_file)

    # print(f"Original message: {MESSAGE}")
    pixels = convert_bits_to_image(decoded_message[1:])
    show_picture(pixels)


    prefix_from_message = decoded_message[0]
    if (prefix_from_message == "p"):
        pass

    if (prefix_from_message == "t"):
        # print(f"Decoded message: {decoded_message[1:]}")
        pass
    
    samples_to_plot = int( 0.1 * SAMPLE_RATE)  
    # plot_debug(time_array, modulated, envelope, bits, energy, samples_to_plot)
