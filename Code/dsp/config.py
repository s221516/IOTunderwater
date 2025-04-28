
 

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

MIC_INDEX = 2 # Mathias, 1 Morten
USE_ESP = True
PLOT_PREAMBLE_CORRELATION = False
PATH_TO_WAV_FILE = "Code/dsp/data/testing_and_logging_recording.wav"
FILE_NAME_DATA_TESTS = "Received_data_for_tests.csv"
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
dark_horse_lyrics = """AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBBCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"""

if __name__ == "__main__":
    import scipy.signal as signal
    
    messages = "_me_w"
    bin_msg = string_to_bin_array(messages)
    autocorrelate = signal.correlate(BINARY_BARKER, bin_msg, mode='same')

    print(autocorrelate)
