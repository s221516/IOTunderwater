
import numpy as np
import scipy.signal as signal

def hamming_distance(received, expected):
    """Computes Hamming distance between received bits and expected bits."""
    if len(received) != len(expected):
        return
    else:
        return sum(r != e for r, e in zip(received, expected))

def bytes_to_bin_array(byte_array):
    """Converts a byte array into a binary array."""
    bin_array = []
    for byte in byte_array:
        bin_array.extend([int(bit) for bit in format(byte, "08b")])
    return bin_array

def string_to_bin_array(string):
    """Converts a string into a binary array."""
    byte_array = string.encode("utf-8")
    return bytes_to_bin_array(byte_array)

def set_bitrate(value):
    global BIT_RATE
    BIT_RATE = value

def set_carrierfreq(value):
    global CARRIER_FREQ
    CARRIER_FREQ = value

# TRANSMITTER_PORT = "COM11"
TRANSMITTER_PORT = "/dev/cu.usbserial-0232D158"
MIC_INDEX = 2 # Mathias, 1 Morten
USE_ESP = False
SAMPLE_RATE = 96000   # this capped by the soundcard, therefore, this is non-changeable

BIT_RATE = 500
CARRIER_FREQ = 11000 
SAMPLES_PER_SYMBOL = int(SAMPLE_RATE / BIT_RATE)
CUT_OFF_FREQ = (CARRIER_FREQ + BIT_RATE) // 2

REP_ESP = 5
# BINARY_BARKER = [1, 1, 1, 0, 0, 1, 0]
# BINARY_BARKER = [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
BINARY_BARKER = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
APPLY_BAKER_PREAMBLE = True
PLOT_PREAMBLE_CORRELATION = False
PATH_TO_WAV_FILE = "Code/dsp/data/testing_and_logging_recording.wav"
# FILE_NAME_DATA_TESTS = "Received_data_for_tests.csv"
# FILE_NAME_DATA_TESTS = "5m_dist_10kHz_unique_payloads.csv"
# FILE_NAME_DATA_TESTS = "1m_distance_payload_barker_similarity_impact.csv"
# FILE_NAME_DATA_TESTS = "1m_distance_carrier_freq_sg_vpp_variable.csv"
# FILE_NAME_DATA_TESTS = "testing_esp.csv"
# FILE_NAME_DATA_TESTS = "spectrography_analysis_esp_sg_without_speaker.csv"
# FILE_NAME_DATA_TESTS = "1m_distance_bitrate_and_carrierfreq_combination.csv"
# FILE_NAME_DATA_TESTS = "computing_freq_impulse.csv"
# FILE_NAME_DATA_TESTS = "Max_bitrate_at_different_distances.csv"
# FILE_NAME_DATA_TESTS = "Varying_payload_sizes.csv"
# FILE_NAME_DATA_TESTS = "Random_payloads.csv"
# FILE_NAME_DATA_TESTS = "avg_power_of_rec_signal_purely_for_check_of_interference.csv"
# FILE_NAME_DATA_TESTS = "Average_power_of_received_signal.csv"
# FILE_NAME_DATA_TESTS = "Max_bitrate_at_different_distances_and_best_carrier_freq.csv"
# FILE_NAME_DATA_TESTS = "Conv_encoding_testing.csv"
FILE_NAME_DATA_TESTS = "Signal_generator_simulation_limit_test.csv"

HAMMING_CODING = False
CONVOLUTIONAL_CODING = False

LIST_OF_DATA_BITS = []
if CONVOLUTIONAL_CODING:
    ENCODING = "Convolutional Encoding"
elif HAMMING_CODING:
    ENCODING = "Hamming Encoding"
else:
    ENCODING = "No Encoding"


IS_ID_SPECIFIED = None
IS_ID_SPECIFIED = ["7906ebdb-664b-46bc-96ea-78049d9e182c"]#"b3c2d680-cb83-4ab0-9cf7-a566ae5d360f","91bf3f5b-c1d0-4492-9f9f-d6030520731d","a70d7f0f-f891-4034-9642-8a144b49a59b"]
# IS_ID_SPECIFIED = ["91bf3f5b-c1d0-4492-9f9f-d6030520731d"]
# IS_ID_SPECIFIED = ["948ebdd5-0edd-47e3-ae00-18c75a484194"] # esp32 attenuated 4000 bits
# IS_ID_SPECIFIED = [
#     "c1ac15b6-66f5-40b5-94df-e1dec5e2961b",
#     "84b73c50-4f0b-48c1-a9c9-24f72ceb35e3",
#     "8ef43baf-f66e-4aa5-908b-6a1cf5a8133b",
#     "0f12a127-f7e4-469a-9506-eb84b20f1a8c",
#     "bb10ad5a-1f34-484b-bb68-fa4aa947c852",
#     "797850c6-259f-4926-90be-8f289a3b2511",
#     "e706c9e4-7571-4c78-9ab8-1013c58f49c5",
#     "7c8a6797-f8dc-4c0f-92aa-a6ec1749fe75",
#     "e1bd6a46-30f9-451f-9ef9-7cf5e8d24824",
#     "183a82c7-d906-4187-ae9a-7e485ca54711",
#     "5dae6d2b-24e1-483d-ac06-f295367be322",
#     "77cb942e-a15c-4296-a12f-eeec21153cf3",
#     "fc3a5201-61a6-40c7-b574-22593fcee20e",
#     "97c514a6-6a03-4a92-a9d0-8745f592b835",
#     "a3b4ad69-c966-41d5-acd2-45d24cdac5c0",
#     "99490ad4-df15-44aa-874d-86852528e048",
#     "9b488b5c-9b30-45c4-bae5-35efcec25cf2",
#     "968b5c1e-0b06-4481-be78-6b653c2548b4",
#     "5b4e4f2f-0d67-4d18-adbe-48614c49e3af",
# ]
# IS_ID_SPECIFIED = ["7c3036d6-654a-4b77-8fe2-2cfc6697f732"] # ESP straight into computer
# IS_ID_SPECIFIED = ["190209f4-1a18-48f7-a33f-7058cc78ac3b"] # ESP straight into computer 4000 bits attenuaed
# IS_ID_SPECIFIED = ["8a58d36b-96f7-424e-b9c8-6cad8323b037"]
# IS_ID_SPECIFIED = ["8a58d36b-96f7-424e-b9c8-6cad8323b037"] # bitrate 1000, cf 10000, esp 
if __name__ == "__main__":
    # import scipy.signal as signal
    
    # print(np.arange(50 // 8, 1050 // 8, 50 // 8))
    messages = "]3MH'@@H9&e6W"
    bin_msg = string_to_bin_array(messages)
    autocorrelate = signal.correlate(bin_msg, BINARY_BARKER, mode='same')

    print(autocorrelate)
