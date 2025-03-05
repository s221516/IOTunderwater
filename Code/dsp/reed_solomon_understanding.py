from reedsolo import RSCodec, ReedSolomonError

def hamming_distance(received, expected):
    """Computes Hamming distance between received bits and expected bits."""
    return sum(r != e for r, e in zip(received, expected))

rsc = RSCodec(12)
message = b"hello world suck"
length_of_message = len(message)
string_encode = rsc.encode(message)
print(string_encode[:length_of_message].decode("utf-8"))

message_with_errors = b'hxllx horld suck'

string_decode = rsc.decode(message_with_errors + string_encode[length_of_message:])[0]

print("Send msg:    ", message.decode("utf-8"))
print("Recived msg: ", message_with_errors.decode("utf-8"))
print("Decoded msg: ", string_decode[:length_of_message].decode("utf-8"))
print("Hamming Distance: ", hamming_distance(message.decode("utf-8"), message_with_errors.decode("utf-8")))