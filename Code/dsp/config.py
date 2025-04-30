
 

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


TRANSMITTER_PORT = "COM12"
# TRANSMITTER_PORT = "/dev/cu.usbserial-0232D158"
SAMPLE_RATE = 96000  # this capped by the soundcard, therefore, this is non-changeable

BIT_RATE = 50
CARRIER_FREQ = 6000  #  15200 Hz
SAMPLES_PER_SYMBOL = int(SAMPLE_RATE / BIT_RATE)
CUT_OFF_FREQ = (CARRIER_FREQ + BIT_RATE) // 2  # TODO: check this value

REP_ESP = 5
# BINARY_BARKER = [1, 1, 1, 0, 0, 1, 0]
# BINARY_BARKER = [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
BINARY_BARKER = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
APPLY_BAKER_PREAMBLE = True
APPLY_AVERAGING_PREAMBLE = False

MIC_INDEX = 1 # Mathias, 1 Morten
USE_ESP = True
PLOT_PREAMBLE_CORRELATION = False
PATH_TO_WAV_FILE = "Code/dsp/data/testing_and_logging_recording.wav"
# FILE_NAME_DATA_TESTS = "Received_data_for_tests.csv"
# FILE_NAME_DATA_TESTS = "Average_power_of_received_signal.csv"
# FILE_NAME_DATA_TESTS = "5m_dist_10kHz_unique_payloads.csv"
# FILE_NAME_DATA_TESTS = "1m_distance_payload_barker_similarity_impact.csv"
FILE_NAME_DATA_TESTS = "1m_distance_checking_carrierfreq_esp_and_sg.csv"

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
# IS_ID_SPECIFIED = "559d7bc2-3804-4a16-bda1-612099de4d49" # old id at 27000 hz, preamble found
# IS_ID_SPECIFIED = "f4a36882-d50e-49e6-9bc6-6edc5fab58ec" # old id at 1000 hz, preamble found
# IS_ID_SPECIFIED = "a7ac7c98-b0b8-41c3-a4cd-d7bcc5eacc0a" # old id at 2000 hz, preamble found
# IS_ID_SPECIFIED = "676ca61d-ff32-443e-8351-2ad8faf8abf7" # old id at 3000 hz, preamble found
# IS_ID_SPECIFIED = "6ec233e2-de76-4f8f-b8dc-12ab2fad6b36" # old id at 4000 hz, preamble found
# IS_ID_SPECIFIED = "f9759151-8eaf-4894-b5aa-c50028345151" # old id at 5000 hz, preamble found
# IS_ID_SPECIFIED = "3279c0bd-defd-4758-a7ae-48f8e24ebab2" # old id at 6000 hz, preamble found
# IS_ID_SPECIFIED = "ca368960-dd4d-4ed7-b769-079246b4220a" # old id at 7000 hz, preamble found
# IS_ID_SPECIFIED = "93641711-f3e0-4974-8d73-374f926e072f" # old id at 8000 hz, preamble found
# IS_ID_SPECIFIED = "f557e626-d12a-4141-926a-d6faf856b786" # old id at 9000 hz, preamble found
# IS_ID_SPECIFIED = "7da7bc6b-249c-40f7-975f-956cd56679e4" # old id at 14000 hz, preamble found
# IS_ID_SPECIFIED = "2cc61e5a-6b97-4778-a7d3-3af7fd46d303" 

dark_horse_lyrics = """AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBBCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"""

if __name__ == "__main__":
    import scipy.signal as signal
    
    messages = "_me_w"
    bin_msg = string_to_bin_array(messages)
    autocorrelate = signal.correlate(BINARY_BARKER, bin_msg, mode='same')

    print(autocorrelate)
