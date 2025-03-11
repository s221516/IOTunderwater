# nyquist = 30400
# 31650 / 15200
import numpy as np
SAMPLE_RATE = 96000  # this capped by the soundcard, therefore, this is non-changeable
CARRIER_FREQ = 15200  #  15200 Hz
BIT_RATE = 200   
SAMPLES_PER_SYMBOL = int(SAMPLE_RATE / BIT_RATE)  # TODO investigate this ratio int(SAMPLE_RATE / BIT_RATE)
CUT_OFF_FREQ = (CARRIER_FREQ + BIT_RATE) // 2  # TODO: check this value
THRESHOLD_BINARY_VAL = 170  # defines when a pixel should be black or white when converting from RGB to black-white
# with 200 samples per bit and 0.45 noise amplitude -> about upper limit, we get a valid picture 1/3 times
NOISE_AMPLITUDE = 0.0  # noise
PATH_TO_WAV_FILE = "Code/dsp/data/recording.wav"
PATH_TO_PICTURE = "./data/doge.jpg"
PREAMBLE_BASE = [0,1,0,0,0,1,1,1]
REPETITIONS = 3
PREAMBLE_PATTERN = PREAMBLE_BASE * REPETITIONS

CONVOLUTIONAL_CODING = True
MAKE_NEW_RECORDING = True
SAMPLE_RATE_FOR_WAV_FILE = 44100  # Hz
RECORD_SECONDS = 3


dark_horse_lyrics = """Yeah, y'all know what it is
    Katy Perry
    Juicy J
    Uh-huh, let's rage
    I knew you were
    You were gonna come to me
    And here you are
    But you better choose carefully
    'Cause I, I'm capable of anything
    Of anything and everything
    Make me your Aphrodite
    Make me your one and only
    But don't make me your enemy (enemy)
    Your enemy (your enemy), your enemy
    So you wanna play with magic?
    Boy, you should know what you're fallin' for
    Baby, do you dare to do this?
    'Cause I'm comin' at you like a dark horse (hey)
    Are you ready for, ready for (hey)
    A perfect storm, perfect storm? (Hey, hey)
    'Cause once you're mine, once you're mine (hey, hey, hey, hey)
    (There's no goin' back)
    Mark my words, this love will make you levitate
    Like a bird, like a bird without a cage
    But down to earth if you choose to walk away
    Don't walk away (walk away)
    It's in the palm of your hand now, baby
    It's a yes or a no, no maybe
    So just be sure before you give it all to me
    All to me, give it all to me
    So you wanna play with magic?
    Boy, you should know what you're fallin' for
    Baby, do you dare to do this?
    'Cause I'm comin' at you like a dark horse (hey)
    Are you ready for, ready for (hey)
    A perfect storm, perfect storm? (Hey, hey)
    'Cause once you're mine, once you're mine (hey, hey, hey, hey)
    Yup (Trippy)
    (There's no going back)
    Uh, she's a beast (beast)
    I call her Karma (come back)
    She eat your heart out like Jeffrey Dahmer (woo)
    Be careful, try not to lead her on
    Shawty heart is on steroids
    'Cause her love is so strong
    You might fall in love when you meet her (meet her)
    If you get the chance, you better keep her (keep her)
    She's sweet as pie, but if you break her heart
    She turn cold as a freezer (freezer)
    That fairy tale ending with a knight in shining armor
    She can be my Sleeping Beauty
    I'm gon' put her in a coma (woo)
    Damn, I think I love her
    Shawty so bad, I'm sprung and I don't care
    She ride me like a roller coaster
    Turn the bedroom into a fair (a fair)
    Her love is like a drug
    I was tryna hit it and quit it
    But lil' mama so dope
    I messed around and got addicted
    So you wanna play with magic?
    Boy, you should know what you're fallin' for (you should know)
    Baby, do you dare to do this?
    'Cause I'm comin' at you like a dark horse (like a dark horse)
    Are you ready for, ready for (ready for)
    A perfect storm, perfect storm? (A perfect storm)
    'Cause once you're mine, once you're mine (mine)
    (There's no going back)"""


# with open("picture_in_binary.txt", "r") as file:
#     picture_in_binary = file.read()

all_letters = "the quick brown fox jumps over the lazy dog while vexd zebras fight for joy! @#$%^&()_+[]{}|;:,.<>/?~` \ The 5 big oxen love quick daft zebras & dogs.>*"
small_test = "This is: 14"
A = "AAAAAAA"

# picture_in_binary_with_prefix = "p" + picture_in_binary


MESSAGE = small_test


PORT = "COM3"


def test():

    bits = [1,0,1,0, 1,1,1,0, 1,0,1,0]    
    pattern_length = 4
    for i in range(0, len(bits) - (3 * pattern_length) + 1, 1):
        avg_pattern = (bits[i:i + pattern_length] + bits[i + pattern_length: i + (2 * pattern_length)] + bits[i + (2 * pattern_length) :  i + (3 * pattern_length)])
        print(avg_pattern)
        for i in range(len(avg_pattern)):
            if avg_pattern[i] > 0.5:
                avg_pattern[i] = 1
            else:
                avg_pattern[i] = 0
    
    print(avg_pattern)


def process_bits(bits, pattern_length):
    avg_pattern = []

    # Compute element-wise average
    for i in range(len(bits) - 2 * pattern_length):
        avg_value = (bits[i] + bits[i + pattern_length] + bits[i + 2 * pattern_length]) / 3
        avg_pattern.append(avg_value)

    # Convert values to 1 or 0
    for i in range(len(avg_pattern)):
        if avg_pattern[i] > 0.5:
            avg_pattern[i] = 1
        else:
            avg_pattern[i] = 0

    return avg_pattern

# Example usage
bits = [1,0,0,0, 1,1,1,0, 1,0,1,1]   
pattern_length = 4
result = process_bits(bits, pattern_length)
print(result)
