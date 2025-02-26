from reedsolo import RSCodec, ReedSolomonError

rsc = RSCodec(10)
string_encode = rsc.encode(b"hello world")
print(string_encode)

string_decode = rsc.decode(b'hxllx horld\xed%T\xc4\xfd\xfd\x89\xf3\xa8\xaa')[0]
print(string_decode)