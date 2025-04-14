import time
import config


class MessageSender:
    def __init__(self, use_esp=True):
        """Initialize the message sender"""
        self.use_esp = use_esp

    def transmit_message(self, message):
        """Transmit a message using either ESP32 or signal generator"""
        try:
            if self.use_esp:
                import esp32test

                print(f"Transmitting '{message}' via ESP32...")
                esp32test.transmit_to_esp32(
                    message, config.CARRIER_FREQ, config.BIT_RATE
                )
            else:
                from transmitterPhysical import transmitPhysical, stopTransmission

                print(f"Transmitting '{message}' via signal generator...")
                transmitPhysical(message, config.CARRIER_FREQ, config.BIT_RATE)
                time.sleep(2)  # Allow time for transmission
                stopTransmission()

            print("Transmission complete")
            return True
        except Exception as e:
            print(f"Transmission error: {e}")
            return False

    def run_interactive_mode(self):
        """Run the interactive command interface"""
        print("\n--- INTERACTIVE SENDER ---")
        print("Enter messages to transmit or commands:")
        print("  'exit' to quit")
        print("  'esp' to switch to ESP32 transmitter")
        print("  'sg' to switch to signal generator")
        print("  'cf=<freq>' to set carrier frequency (e.g., 'cf=6000')")
        print("  'br=<rate>' to set bit rate (e.g., 'br=100')")

        try:
            while True:
                user_input = input("\n> ")

                if not user_input.strip():
                    continue

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


if __name__ == "__main__":
    # Default to ESP32 transmission
    sender = MessageSender(use_esp=True)
    sender.run_interactive_mode()
