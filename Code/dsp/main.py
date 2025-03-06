from transmitterClass import Transmitter


def main():
    message = "AA"
    transmitter = Transmitter(message)
    transmitter.transmit()

    time.sleep(1)

    # Non-Coherent demodulation
    receiver_non_coherent = NonCoherentReceiver.from_wav_file(PATH_TO_WAV_FILE)
    message_nc, debug_nc = receiver_non_coherent.decode()
    print(f"Non-Coherent Decoded: {message_nc}")

    # Coherent demodulation
    receiver_coherent = CoherentReceiver.from_wav_file(PATH_TO_WAV_FILE)
    message_c, debug_c = receiver_coherent.decode()
    print(f"Coherent Decoded: {message_c}")
    plot_demodulation_steps(receiver_coherent, debug_c, "Coherent")

if __name__ == "__main__":
    main()
