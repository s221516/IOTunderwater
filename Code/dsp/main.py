from transmitter.transmitterClass import transmitPhysical, transmitVirtual
from receiver.receiverClass import NonCoherentReceiver, CoherentReceiver
from config_values import PATH_TO_WAV_FILE, SAMPLE_RATE, MESSAGE, RECORD_SECONDS
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from receiver.record_audio import create_wav_file_from_recording

# Modified main function


def main():
    # Start recording
    create_wav_file_from_recording(RECORD_SECONDS)
    
    #Start transmission
    transmitPhysical(MESSAGE)
    #transmitVirtual(message)

    # time.sleep(5)
    # # Non-Coherent demodulation
    # receiver_non_coherent = NonCoherentReceiver.from_wav_file(PATH_TO_WAV_FILE)
    # message_nc, debug_nc = receiver_non_coherent.decode()
    # print(f"Non-Coherent Decoded: {message_nc}")
    # receiver_non_coherent.plot_simulation_steps()

    # # Coherent demodulation
    # receiver_coherent = CoherentReceiver.from_wav_file(PATH_TO_WAV_FILE)
    # message_c, debug_c = receiver_coherent.decode()
    # print(f"Coherent Decoded: {message_c}")
    # receiver_coherent.plot_simulation_steps()

if __name__ == "__main__":
    main()
