
 

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


TRANSMITTER_PORT = "COM11"
# TRANSMITTER_PORT = "/dev/cu.usbserial-0232D158"
MIC_INDEX = 1 # Mathias, 1 Morten
USE_ESP = False
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
FILE_NAME_DATA_TESTS = "1m_distance_bitrate_and_carrierfreq_combination.csv"
# FILE_NAME_DATA_TESTS = "computing_freq_impulse.csv"

HAMMING_CODING = False
CONVOLUTIONAL_CODING = False

LIST_OF_DATA_BITS = []
if CONVOLUTIONAL_CODING:
    ENCODING = "Convolutional Encoding"
elif HAMMING_CODING:
    ENCODING = "Hamming Encoding"
else:
    ENCODING = "No Encoding"


#IS_ID_SPECIFIED = None
IS_ID_SPECIFIED = "106158f2-e357-49eb-be28-e43a80eaa551"
#IS_ID_SPECIFIED = ["a0c4f3b2-1d8e-4f5b-9a6c-7d0e1f2a3b4c"]

dark_horse_lyrics = """AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBBCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"""

if __name__ == "__main__":
    import scipy.signal as signal
    
    messages = "_me_w"
    bin_msg = string_to_bin_array(messages)
    autocorrelate = signal.correlate(BINARY_BARKER, bin_msg, mode='same')

    print(autocorrelate)
