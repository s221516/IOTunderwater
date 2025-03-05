import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
from config_values import (
    CARRIER_FREQ,
    PATH_TO_WAV_FILE,
    SAMPLE_RATE,
    SAMPLE_RATE_FOR_WAV_FILE,
    SAMPLES_PER_SYMBOL,
)

np.set_printoptions(precision=4, suppress=True)


# returns two arrays, square_wave and the time_array
def make_square_wave(message: str):
    message_binary = "".join(format(ord(i), "08b") for i in message)
    print("Message binary:", message_binary)
    square_wave = []
    for bit in message_binary:
        square_wave += [int(bit)] * SAMPLES_PER_SYMBOL
    # Ensure even sampling alignment
    if len(square_wave) % SAMPLES_PER_SYMBOL != 0:
        square_wave += [0] * (
            SAMPLES_PER_SYMBOL - len(square_wave) % SAMPLES_PER_SYMBOL
        )
    duration = len(square_wave) / SAMPLE_RATE
    time_array = np.linspace(0, duration, len(square_wave))

    return np.array(square_wave), time_array


def make_carrier_wave(time_array) -> np.array:
    # NOTE: to add noise check the bottom of freq domain section of pysdr
    carrier_wave = np.sin(2 * np.pi * CARRIER_FREQ * time_array)

    return carrier_wave


def plot_waveforms(square_wave, carrier_wave, modulated_wave, time_array):
    plt.figure(figsize=(12, 6))

    # Plot the square wave
    plt.subplot(3, 1, 1)
    plt.plot(time_array, square_wave, label="Square Wave")
    plt.title("Square Wave")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # Plot the carrier wave
    plt.subplot(3, 1, 2)
    plt.plot(time_array, carrier_wave, label="Carrier Wave", color="orange")
    plt.title("Carrier Wave")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # Plot the carrier wave
    plt.subplot(3, 1, 3)
    plt.plot(time_array, modulated_wave, label="Modulated Wave", color="pink")
    plt.title("Modulated Wave")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def create_modulated_wave(square_wave: np.array, carrier_wave: np.array) -> np.array:
    # For AM, we need to ensure the square wave is offset to be positive
    # Convert from [0,1] to [0.5,1] to maintain carrier amplitude
    normalized_square_wave = 0.5 + 0.5 * square_wave

    # Multiply the carrier by the modulating signal
    return normalized_square_wave * carrier_wave


def write_to_wav_file(modulated_wave: np.array):
    wav.write(PATH_TO_WAV_FILE, SAMPLE_RATE_FOR_WAV_FILE, modulated_wave)


if __name__ == "__main__":
    katy = """Yeah, y'all know what it is
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
    all_letters = "the quick brown fox jumps over the lazy dog while vexd zebras fight for joy! @#$%^&()_+[]{}|;:,.<>/?~` \ The 5 big oxen love quick daft zebras & dogs.>*"
    a_ = "20000000000 dollars"
    square_wave, time_array = make_square_wave(a_)
    carrier_wave = make_carrier_wave(time_array)
    print(len(time_array), np.shape(time_array))
    modulated_wave = create_modulated_wave(square_wave, carrier_wave)
    write_to_wav_file(modulated_wave)
    plot_waveforms(square_wave, carrier_wave, modulated_wave, time_array)
