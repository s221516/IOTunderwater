import time
import config
from receiver.receiverClass import NonCoherentReceiver
import receiver.record_audio as rc


def record_audio():
    id = "test_recording"
    rc.create_wav_file_from_recording(5, id)
    
    

def show_signal():
    id = "test_recording"
    receiver = NonCoherentReceiver(id=id, band_pass=False)
    
    receiver.plot_signal()

if __name__ == "__main__":
    print("Recording audio...")
    record_audio()
    time.sleep(2) 
    print("Showing signal...")
    show_signal()