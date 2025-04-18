import serial

def initPort(portName):
    try:
        # Initialize serial port
        ser = serial.Serial(
            port=portName,  
            baudrate=57600,  # Default baudrate
            bytesize=serial.EIGHTBITS,  # 8 bits per byte
            parity=serial.PARITY_NONE,  # No parity
            stopbits=serial.STOPBITS_ONE,  # 1 stop bit
            timeout=1,  # 1 second timeout for read/write operations
            dsrdtr=True,  # Enable DSR/DTR hardware handshaking
            rtscts=False,  # Disable RTS/CTS flow control
            xonxoff=False  # Disable software flow control (XON/XOFF)
        )
        
        # Check if the port was opened successfully
        if ser.is_open:
            # print(f"Port {portName} opened successfully.")
            pass
        else:
            print(f"Failed to open {portName}.")
            exit()

        # Return the serial connection object
        return ser

    except serial.SerialException as e:
        # Exception handling for serial port issues
        print(f"Failed to open port {portName}: {e}")
        exit()

    except Exception as e:
        # Catch any other exception not related to serial issues
        print(f"An unexpected error occurred: {e}")
        exit()
