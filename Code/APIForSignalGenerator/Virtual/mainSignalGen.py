import serial
import time

from initPorts import initPort
def main():
    ser = initPort('COM8')
    message = "IDN?"
    ser.write(message.encode('utf-8'))

    time.sleep(10)


    if ser.in_waiting > 0:
        response = ser.read(ser.in_waiting).decode('utf-8')
        print(f"Received: {response.strip()}\n")
    else:
        print("No response received\n")

if __name__ == "__main__":
    main()