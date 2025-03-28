import bluetooth
import time


#This returns MAC address that i used to connect
def find_device_by_name(target_name):
    # Start scanning for nearby Bluetooth devices
    print("Scanning for nearby Bluetooth devices...")
    nearby_devices = bluetooth.discover_devices(lookup_names=True, lookup_uuids=True)
    
    for addr, name in nearby_devices:
        print(f"Found device: {name} - {addr}")
        
        # Check if the name matches the target name
        if target_name in name:
            print(f"Found target device: {target_name} with MAC address {addr}")
            return addr
    return None

#Connects to ESP32 using the MAC address
def connect_to_esp32(mac_address):
    # The RFCOMM port 1 is typically used for SPP connections
    port = 1  # Common port for SPP (Serial Port Profile)
    
    # Try to connect to the ESP32 Bluetooth device at the specified MAC address
    try:
        print(f"Attempting to connect to ESP32 at {mac_address}...")
        
        # Create a Bluetooth socket using RFCOMM
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        
        # Connect to the ESP32 device using its MAC address and the SPP port
        sock.connect((mac_address, port))
        print("Connected to ESP32 Bluetooth server!")
        
        return sock
        
    
    except bluetooth.btcommon.BluetoothError as err:
        print(f"Bluetooth error: {err}")
    except Exception as e:
        print(f"Error: {e}")


#Transmit the message to the ESP32, which then trasmits using the speaker
def transmit(sock, message):
    
    sock.send(message)
    print(f"Sent message: {message}")
    
    # Receive the response from ESP32
    response = sock.recv(1024)  # Adjust buffer size as needed
    print(f"Received from ESP32: {response.decode('utf-8')}")


def main():

    print("connecting to ESP32 bluetooth")
    while True:
        mac_address = find_device_by_name("ESP32_Bluetooth")
        if mac_address:
            sock = connect_to_esp32(mac_address)
            break
        else:
            print("Device not found. Retrying...")
            time.sleep(1)
    
    print("Connected to ESP32 Bluetooth server!")
    while True:
        try:
            message = input("Enter your message: ")
            transmit(sock, message)
            
            #response = receive()
            #display(response)

        finally:
            sock.close()
            print("Connection closed.")

