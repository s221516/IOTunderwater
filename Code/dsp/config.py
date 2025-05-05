
 

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
MIC_INDEX = 1 # Mathias, 1 Morten
USE_ESP = True
SAMPLE_RATE = 96000  # this capped by the soundcard, therefore, this is non-changeable

BIT_RATE = 50
CARRIER_FREQ = 1000 
SAMPLES_PER_SYMBOL = int(SAMPLE_RATE / BIT_RATE)
CUT_OFF_FREQ = (CARRIER_FREQ + BIT_RATE) // 2

REP_ESP = 8
# BINARY_BARKER = [1, 1, 1, 0, 0, 1, 0]
# BINARY_BARKER = [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
BINARY_BARKER = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
APPLY_BAKER_PREAMBLE = True
APPLY_AVERAGING_PREAMBLE = False
PLOT_PREAMBLE_CORRELATION = False
PATH_TO_WAV_FILE = "Code/dsp/data/testing_and_logging_recording.wav"
# FILE_NAME_DATA_TESTS = "Received_data_for_tests.csv"
# FILE_NAME_DATA_TESTS = "Average_power_of_received_signal.csv"
# FILE_NAME_DATA_TESTS = "5m_dist_10kHz_unique_payloads.csv"
# FILE_NAME_DATA_TESTS = "1m_distance_payload_barker_similarity_impact.csv"
# FILE_NAME_DATA_TESTS = "1m_distance_carrier_freq_sg_vpp_variable.csv"
# FILE_NAME_DATA_TESTS = "testing_esp.csv"
# FILE_NAME_DATA_TESTS = "spectrography_analysis_esp_sg_without_speaker.csv"
# FILE_NAME_DATA_TESTS = "1m_distance_bitrate_and_carrierfreq_combination.csv"
# FILE_NAME_DATA_TESTS = "computing_freq_impulse.csv"
# FILE_NAME_DATA_TESTS = "Max_bitrate_at_different_distances.csv"
# FILE_NAME_DATA_TESTS = "Max_bitrate_at_different_distances_and_best_carrier_freq.csv"
FILE_NAME_DATA_TESTS = "1m_distance_payload_barker_similarity_impact.csv"

HAMMING_CODING = True
CONVOLUTIONAL_CODING = True

LIST_OF_DATA_BITS = []
if CONVOLUTIONAL_CODING:
    ENCODING = "Convolutional Encoding"
elif HAMMING_CODING:
    ENCODING = "Hamming Encoding"
else:
    ENCODING = "No Encoding"


IS_ID_SPECIFIED = None
IS_ID_SPECIFIED = ["645a52b8-f288-4eff-aeba-91434114baa5"]
# IS_ID_SPECIFIED = [
#     "2f61767c-fd61-4bce-8b9d-31d74d0f6a03",
#     "894d52ee-80e4-41ca-afde-044726f3cafe",
#     "b10bfe28-54d6-42f0-ad16-cfbd86cd623c",
# ]
# IS_ID_SPECIFIED = ["77973a98-2385-4e4f-92a2-904a48a1b0fb"]
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


if __name__ == "__main__":
    import scipy.signal as signal
    
    messages = "_me_w"
    bin_msg = string_to_bin_array(messages)
    autocorrelate = signal.correlate(BINARY_BARKER, bin_msg, mode='same')

    print(autocorrelate)
