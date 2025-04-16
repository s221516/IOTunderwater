import time
from datetime import datetime
import config
import threading


class MessageSender(threading.Thread):
    def __init__(self, shared_state, use_esp=True):
        super().__init__(name="MessageSenderThread")
        """Initialize the message sender"""
        self.use_esp = use_esp
        self.shared_state = shared_state

    def transmit_message(self, message):
        self.shared_state["is_transmitting"] = True

        # print("Transmitting: ", self.shared_state["is_transmitting"])

        # NOTE: Make this less reliant on config.SOME_VALUE
        try:
            if self.use_esp:
                import esp32test

                esp32test.transmit_to_esp32(
                    message, config.CARRIER_FREQ, config.BIT_RATE
                )
            else:
                from transmitterPhysical import transmitPhysical, stopTransmission

                transmitPhysical(message, config.CARRIER_FREQ, config.BIT_RATE)
                stopTransmission()

            # Wait for the transmission to complete
            len_of_bits = len(message) * 8 + 13
            transmission_time = (
                config.REP_ESP * (len_of_bits / config.BIT_RATE)
                + (1 / 240000) * 1000000
            )
            time.sleep(transmission_time)

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
