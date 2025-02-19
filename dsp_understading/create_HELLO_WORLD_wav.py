import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav

# Define parameters
sample_rate = 44100  # 44.1 kHz standard audio sampling rate
carrier_freq = 10000  # Hz
bit_rate = 1000  # Bits per second (slow for clarity)
duration_per_bit = 1 / bit_rate  # seconds per bit

# Convert "HELLO WORLD" to binary ASCII
text = """Do you ever feel like a plastic bag
Drifting through the wind
Wanting to start again?
Do you ever feel, feel so paper-thin
Like a house of cards, one blow from caving in?

Do you ever feel already buried deep?
Six feet under screams but no one seems to hear a thing
Do you know that there's still a chance for you
'Cause there's a spark in you?

You just gotta ignite the light and let it shine
Just own the night like the 4th of July

'Cause, baby, you're a firework
Come on, show 'em what you're worth
Make 'em go, "Ah, ah, ah"
As you shoot across the sky

Baby, you're a firework
Come on, let your colors burst
Make 'em go, "Ah, ah, ah"
You're gonna leave 'em all in awe, awe, awe

You don't have to feel like a wasted space
You're original, cannot be replaced
If you only knew what the future holds
After a hurricane comes a rainbow

Maybe a reason why all the doors are closed
So you could open one that leads you to the perfect road
Like a lightning bolt your heart will glow
And when it's time you'll know

You just gotta ignite the light and let it shine
Just own the night like the 4th of July

'Cause, baby, you're a firework
Come on, show 'em what you're worth
Make 'em go, "Ah, ah, ah"
As you shoot across the sky

Baby, you're a firework
Come on, let your colors burst
Make 'em go, "Ah, ah, ah"
You're gonna leave 'em all in awe, awe, awe

Boom, boom, boom
Even brighter than the moon, moon, moon
It's always been inside of you, you, you
And now it's time to let it through, -ough, -ough

'Cause, baby, you're a firework
Come on, show 'em what you're worth
Make 'em go, "Ah, ah, ah"
As you shoot across the sky

Baby, you're a firework
Come on, let your colors burst
Make 'em go, "Ah, ah, ah"
You're gonna leave 'em all in awe, awe, awe

Boom, boom, boom
Even brighter than the moon, moon, moon
Boom, boom, boom
Even brighter than the moon, moon, moon"""

# text = "The quick brown fox jumps over the lazy dog while vexd zebras fight for joy! @#$%^&()_+[]|;:,.<>/?~` \ The 5 big oxen love quick daft zebras & dogs.>*"
binary_message = "".join(format(ord(c), "08b") for c in text)  # 8-bit ASCII encoding
print(f"Binary Representation: {binary_message}")

# Generate time array
total_duration = len(binary_message) * duration_per_bit
t = np.linspace(0, total_duration, int(sample_rate * total_duration), endpoint=False)

# Create the square wave modulation signal
modulation_signal = np.repeat(
    [int(b) for b in binary_message], int(sample_rate * duration_per_bit)
)
modulation_signal = np.pad(
    modulation_signal, (0, len(t) - len(modulation_signal)), "constant"
)

# Normalize modulation signal to 0.3 - 1 (so carrier is not fully suppressed at 0s)
modulation_signal = (
    0.3 + 0.7 * modulation_signal
)  # Adjust amplitude between 0.3 (low) and 1 (high)

# Create the carrier wave (10 Hz sine wave)
carrier_wave = np.sin(2 * np.pi * carrier_freq * t)

# Apply amplitude modulation (AM)
modulated_wave = modulation_signal * carrier_wave

# Normalize to int16 range and save
modulated_wave = (modulated_wave * 32767).astype(
    np.int16
)  # Convert to 16-bit PCM format
wav.write("hello_world_am.wav", sample_rate, modulated_wave)

# Plot the modulation signal and modulated wave
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t[: int(2 * sample_rate)], modulation_signal[: int(2 * sample_rate)], "r")
plt.title("Square Wave Modulation Signal (First 2 Seconds)")
plt.subplot(2, 1, 2)
plt.plot(t[: int(2 * sample_rate)], modulated_wave[: int(2 * sample_rate)], "b")
plt.title("AM Modulated Signal (First 2 Seconds)")
plt.tight_layout()
plt.show()
