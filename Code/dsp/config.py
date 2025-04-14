def message_toBitArray(message: str):
    message_binary = "".join(format(ord(i), "08b") for i in message)
    # TODO: determine the exact number of samples per bit that makes sense in relation to our sample rate
    # and bits per second and also how this is done in the signal generator

    square_wave = []
    for bit in message_binary:
        if bit == "0":
            square_wave += [0]
        elif bit == "1":
            square_wave += [1]

    return square_wave


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


PORT = "COM11"
SAMPLE_RATE = 96000  # this capped by the soundcard, therefore, this is non-changeable

BIT_RATE = 100
CARRIER_FREQ = 6000  #  15200 Hz
SAMPLES_PER_SYMBOL = int(SAMPLE_RATE / BIT_RATE)
CUT_OFF_FREQ = (CARRIER_FREQ + BIT_RATE) // 2  # TODO: check this value

PREAMBLE_BASE = message_toBitArray("G")
REPETITIONS = 3
PREAMBLE_PATTERN = PREAMBLE_BASE * REPETITIONS
# BINARY_BARKER = [1, 1, 1, 0, 0, 1, 0]
# BINARY_BARKER = [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
BINARY_BARKER = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
APPLY_BAKER_PREAMBLE = True
APPLY_AVERAGING_PREAMBLE = False

RECORD_FOR_LOOP_TESTING = False

PLOT_PREAMBLE_CORRELATION = False

if RECORD_FOR_LOOP_TESTING:
    PATH_TO_WAV_FILE = "Code/dsp/data/testing_and_logging_recording.wav"
else:
    PATH_TO_WAV_FILE = "Code/dsp/data/chatting_recording.wav"

STAGE_1 = False

HAMMING_CODING = False
CONVOLUTIONAL_CODING = False

if CONVOLUTIONAL_CODING:
    ENCODING = "Convolutional Encoding"
elif HAMMING_CODING:
    ENCODING = "Hamming Encoding"
else:
    ENCODING = "No Encoding"


dark_horse_lyrics = """AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBBCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"""
