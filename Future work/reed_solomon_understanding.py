from reedsolo import RSCodec, ReedSolomonError

def hamming_distance(received, expected):
    """Computes Hamming distance between received bits and expected bits."""
    return sum(r != e for r, e in zip(received, expected))

def compute_max_errors_correctable(n, k):
    """Computes the maximum number of errors that can be corrected."""
    return (n - k) // 2

def bytes_to_bin_array(byte_array):
    """Converts a byte array into a binary array."""
    bin_array = []
    for byte in byte_array:
        bin_array.extend([int(bit) for bit in format(byte, '08b')])
    return bin_array

num_of_error_correcting_symbols = 2
rsc = RSCodec(num_of_error_correcting_symbols)
message             = b"44"
message_with_errors = b'40'

length_of_message = len(message)
string_encode = rsc.encode(message)
print("Byte encoded array: ", string_encode)
print(len(bytes_to_bin_array(string_encode)))
print("Bit encoded array: ", bytes_to_bin_array(string_encode))

string_decode = rsc.decode(message_with_errors + string_encode[length_of_message:])[0]

print("Send msg:    ", message.decode("utf-8"))
print("Recived msg: ", message_with_errors.decode("utf-8"))
print("Decoded msg: ", string_decode[:length_of_message].decode("utf-8"))
print("Hamming Distance: ", hamming_distance(message.decode("utf-8"), message_with_errors.decode("utf-8")))
n = length_of_message + num_of_error_correcting_symbols
k = length_of_message
print("Max no of correctable errors: ", compute_max_errors_correctable(n, k))