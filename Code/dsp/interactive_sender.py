import time
from datetime import datetime
import config


class MessageSender:
    def __init__(self, use_esp=True):
        """Initialize the message sender"""
        self.use_esp = use_esp

    def get_is_transmitting():
        return AM_I_TRANSMITTING

    def transmit_message(self, message):
        """Transmit a message using either ESP32 or signal generator"""
        global AM_I_TRANSMITTING
        AM_I_TRANSMITTING = True
        transmission_start = datetime.now()
        print("Transmitting: ", AM_I_TRANSMITTING)

        # NOTE: Make this less reliant on config.SOME_VALUE        
        try:
            if self.use_esp:
                import esp32test
                esp32test.transmit_to_esp32(message, config.CARRIER_FREQ, config.BIT_RATE)
            else:
                from transmitterPhysical import transmitPhysical, stopTransmission
                transmitPhysical(message, config.CARRIER_FREQ, config.BIT_RATE)
                stopTransmission()

            # NOTE: Compute the actual time is takes to transmit something - this is also needed in esp32test
            while (datetime.now() - transmission_start).total_seconds() < 10:
                pass  # Non-blocking wait
                
            return True
        except Exception as e:
            print(f"Transmission error: {e}")
            return False
        finally:
            AM_I_TRANSMITTING = False
            print("Transmitting: ", AM_I_TRANSMITTING)

    def run_interactive_mode(self):
        """Run the interactive command interface"""
        print("\n--- INTERACTIVE SENDER ---")
        print("Enter messages to transmit or commands:")
        print("  'exit' to quit")
        print("  'esp' to switch to ESP32 transmitter")
        print("  'sg' to switch to signal generator")
        print("  'cf=<freq>' to set carrier frequency (e.g., 'cf=6000')")
        print("  'br=<rate>' to set bit rate (e.g., 'br=100') \n")

        try:
            print("Write a message below!")
            while True:
                user_input = input("You: ")

                if user_input.lower() == "exit":
                    break

                elif user_input.lower() == "esp":
                    self.use_esp = True
                    print("Switched to ESP32 transmitter")

                elif user_input.lower() == "sg":
                    self.use_esp = False
                    print("Switched to signal generator")

                elif user_input.lower().startswith("cf="):
                    try:
                        freq = int(user_input[3:])
                        config.CARRIER_FREQ = freq
                        print(f"Carrier frequency set to {freq} Hz")
                    except ValueError:
                        print("Invalid frequency format. Use 'cf=6000' format.")

                elif user_input.lower().startswith("br="):
                    try:
                        rate = int(user_input[3:])
                        config.BIT_RATE = rate
                        print(f"Bit rate set to {rate} bps")
                    except ValueError:
                        print("Invalid bit rate format. Use 'br=100' format.")

                else:
                    # Transmit the message
                    self.transmit_message(user_input)

        except KeyboardInterrupt:
            print("\nExiting interactive mode...")

