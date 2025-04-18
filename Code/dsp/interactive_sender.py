import time
import config
import threading


class MessageSender(threading.Thread):
    def __init__(self, shared_state):
        super().__init__(name="MessageSenderThread")
        """Initialize the message sender"""
        self.shared_state = shared_state

    def transmit_message(self, message):
        # For 1 computer setup and testing set it to False otherwise True
        self.shared_state["is_transmitting"] = True

        # print("Transmitting: ", self.shared_state["is_transmitting"])

        # NOTE: Make this less reliant on config.SOME_VALUE
        try:
            if config.USE_ESP:
                import esp32test

                esp32test.transmit_to_esp32(
                    message, config.CARRIER_FREQ, config.BIT_RATE
                )
            else:
                from transmitterPhysical import transmitPhysical, stopTransmission

                transmitPhysical(message, config.CARRIER_FREQ, config.BIT_RATE)

            len_of_bits = len(message) * 8 + 13

            transmission_time = (
                config.REP_ESP * (len_of_bits / config.BIT_RATE)
                + (1 / 240000) * 1000000
            )
            # Wait for the transmission to complete
            time.sleep(transmission_time)
            
            if not config.USE_ESP:
                stopTransmission()
            else:
                esp32test.send_command("STOP")

        except Exception as e:
            print(f"Transmission error: {e}")
        finally:
            self.shared_state["is_transmitting"] = False
            # print("Transmitting: ", self.shared_state["is_transmitting"])

    def run(self):
        while True:
            if self.shared_state["msg"] is not None:
                self.transmit_message(self.shared_state["msg"])
                self.shared_state["msg"] = None
            time.sleep(1)
