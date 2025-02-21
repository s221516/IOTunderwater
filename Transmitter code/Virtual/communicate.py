import serial
import Ports
import time

def initPort(portName):
    try:

        ser = serial.Serial(
            port=portName,  # Replace with the correct port name (e.g., "COM3" or "/dev/ttyUSB0")
            baudrate= 57600,  # 300, 600, 1200, 2400, 4800, 9600, 19200, 38400, 57600 (factory setting), 115200
            bytesize=serial.EIGHTBITS,  # 8   \  7   \  7  (databits is connected with parity below)
            parity=serial.PARITY_NONE,  #None \ Even \ Odd
            stopbits=serial.STOPBITS_ONE,  # 1 stop bit (fixed for the generator)
            #python always assumes a start bit - this might be false, since source is chat.
            timeout=1,  # How long the program will wait for data / how long it will wait to write data (?)
            dsrdtr=True,  # Enable DTR/DSR hardware handshake (factory default)
            rtscts=False,  # Disable RTS/CTS (not used by default)
            xonxoff=False  # Disable XON/XOFF (not used by default)
        )

        return ser
    except serial.SerialException as e:
        print(f"Failed to open port: {e}")
        exit()
def communicateVirtual(message):
    try:
        Ports.ser_write.write(message.encode('utf-8'))
        print(f"Sent: {message.strip()}")

        # Wait for response on COM4
        time.sleep(1)  # Allow time for data to transmit
        if Ports.ser_read.in_waiting > 0:
            response = Ports.ser_read.read(Ports.ser_read.in_waiting).decode('utf-8')
            #print(f"Received: {response.strip()}\n")
        else:
            print("No response received on COM4.\n")

    finally:
        pass
def testConnectionVirtual():

    t = 3

    print("\nCreating square carrier wave")
    communicateVirtual("FUNC SQU")
    communicateVirtual("FREQ 5000")
    communicateVirtual("VOLT 3.0")
    communicateVirtual("VOLT:OFFS -2.5")

    #TODO maybe like the below instead?
    #communicateVirtual("FUNC SQU; FREQ 5000; VOLT 3.0; VOLT:OFFS -2.5 ")

    time.sleep(t)

    print("\nCreating sin carrier wave")
    communicateVirtual("APPL:SIN 5.0E+3, 3.0, -2.5")  # TODO See PAGE 144 to see exactly what happens in the machine

    time.sleep(t)
    print("\nAM SIN wave with interal square wave")

    communicateVirtual("AM: SOUR INT") #or maybe communicateVirtual("AM: INT: FUNC")
    communicateVirtual("AM: INT:FUNC") #
    communicateVirtual("AM:INT:FREQ")  #SET FREQ: Change FREQ. Set the modulating frequency to any value from 2 mHz to 20 kHz using the AM:INT:FREQ command.
    communicateVirtual("AM:DEPT")  # SET FREQ: Change FREQ. Set the modulating frequency to any value from 2 mHz to 20 kHz using the AM:INT:FREQ command.

    time.sleep(t)
    print("\nAM the wave with the arbitrary wave form")

    communicateVirtual("DATA VOLATILE, 0, 0.382, 0.707, 0.924, 1, 0.924, 0.707, 0.382, 0, -0.382, -0.707, -0.924, -1, -0.924, -0.707, -0.382, 0")
    communicateVirtual("FREQ 1")

    time.sleep(t)
    print("\nAM the wave with the arbitrary wave form")
    communicateVirtual("DATA VOLATILE, 0, 0.924, -0.707, -0.924, -1, -0.924")
    communicateVirtual("FREQ 1")


def communicatePhysical(message):
    try:
        Ports.ser.write(message.encode('utf-8'))
        print(f"Sent: {message.strip()}")

    finally:
        pass
def testConnectionPhysical():

    t = 3

    print("\nCreating square carrier wave")
    communicatePhysical("FUNC SQU")
    communicatePhysical("FREQ 5000")
    communicatePhysical("VOLT 3.0")
    communicatePhysical("VOLT:OFFS -2.5")

    #TODO test if the below works
    communicatePhysical("FREQ?") #querying frequency
    response = Ports.ser.readline()  # Read the response (in bytes)
    response_str = response.decode('utf-8').strip()
    print("Recieved" + response_str)

    # TODO maybe like the below instead?
    # communicateVirtual("FUNC SQU; FREQ 5000; VOLT 3.0; VOLT:OFFS -2.5 ")

    time.sleep(t)

    print("\nCreating sin carrier wave")
    communicatePhysical("APPL:SIN 5.0E+3, 3.0, -2.5")  # TODO See PAGE 144 to see exactly what happens in the machine

    time.sleep(t)
    print("\nAM SIN wave with internal square wave")

    communicatePhysical("AM: SOUR INT") #or maybe communicatePhysical("AM: INT: FUNC")
    communicatePhysical("AM: INT:FUNC") #
    communicatePhysical("AM:INT:FREQ")  #SET FREQ: Change FREQ. Set the modulating frequency to any value from 2 mHz to 20 kHz using the AM:INT:FREQ command.
    communicatePhysical("AM:DEPT")  # SET FREQ: Change FREQ. Set the modulating frequency to any value from 2 mHz to 20 kHz using the AM:INT:FREQ command.

    time.sleep(t)
    print("\nAM the wave with the arbitrary wave form")

    communicatePhysical("DATA VOLATILE, 0, 0.382, 0.707, 0.924, 1, 0.924, 0.707, 0.382, 0, -0.382, -0.707, -0.924, -1, -0.924, -0.707, -0.382, 0")
    communicatePhysical("FREQ 1")

    time.sleep(t)
    print("\nAM the wave with the arbitrary wave form")
    communicatePhysical("DATA VOLATILE, 0, 0.924, -0.707, -0.924, -1, -0.924")
    communicatePhysical("FREQ 1")