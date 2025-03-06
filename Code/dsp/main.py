from transmitter.transmitterClass import Transmitter
from receiver.receiverClass import NonCoherentReceiver, CoherentReceiver
from config_values import PATH_TO_WAV_FILE, SAMPLE_RATE, MESSAGE
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Modified main function
def main():
    message = "A"
    transmitter = Transmitter(MESSAGE)
    transmitter.transmit()

    time.sleep(1)

    # Non-Coherent demodulation
    receiver_non_coherent = NonCoherentReceiver.from_wav_file(PATH_TO_WAV_FILE)
    message_nc, debug_nc = receiver_non_coherent.decode()
    print(f"Non-Coherent Decoded: {message_nc}")
    receiver_non_coherent.plot_simulation_steps()

    # Coherent demodulation
    receiver_coherent = CoherentReceiver.from_wav_file(PATH_TO_WAV_FILE)
    message_c, debug_c = receiver_coherent.decode()
    print(f"Coherent Decoded: {message_c}")
    receiver_coherent.plot_simulation_steps()

if __name__ == "__main__":
    main()
