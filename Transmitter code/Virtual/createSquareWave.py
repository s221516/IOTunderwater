


#Make sure this is ascii
def string_to_binary(input_string):
    return ''.join(format(ord(char), '08b') for char in input_string)
def binary_to_squareWave(bit_string, floatPointsPerBit:int):
    squareWave = []
    for i in bit_string:

        if i == "1":
            for j in range(0,floatPointsPerBit):
                squareWave.append(0.5)

        else:
            for j in range(0,floatPointsPerBit):
                squareWave.append(-0.5)

    return squareWave

#test 0's on both sides
def addPaddingToSquareWave(SquareWave):
    targetSize = 50 #65,536 or 16,384 (page 198 and 298 )
    paddingSpots = targetSize - len(SquareWave)
    if paddingSpots < 2: #If we have no padding spots, then its GG - our message is simply to big.
        return 0
    div, mod = divmod(paddingSpots,2) #find how much padding we can make, i guess the more the better - we simply record more.
    startPadding = [1] * div
    stopPadding = [-1] * div
    B = startPadding + SquareWave + stopPadding + [-1] * mod
    return B
def convertWaveToSCPI(wave):
    return 'DATA ' + ', '.join(map(str, wave))

message = "C"
messageBinary = string_to_binary(message)
print(messageBinary)
squareWave = binary_to_squareWave(messageBinary, 5)
print(squareWave)
squareWaveWithPadding = addPaddingToSquareWave(squareWave)
print(squareWaveWithPadding)
SCPIcommand=convertWaveToSCPI(squareWaveWithPadding)
print(SCPIcommand)