
import numpy as np
import scipy.signal as signal

def hamming_distance(received, expected):
    """Computes Hamming distance between received bits and expected bits."""
    if len(received) != len(expected):
        return
    else:
        return sum(r != e for r, e in zip(received, expected))

def bytes_to_bin_array(byte_array):
    """Converts a byte array into a binary array."""
    bin_array = []
    for byte in byte_array:
        bin_array.extend([int(bit) for bit in format(byte, "08b")])
    return bin_array

def string_to_bin_array(string):
    """Converts a string into a binary array."""
    byte_array = string.encode("utf-8")
    return bytes_to_bin_array(byte_array)

def set_bitrate(value):
    global BIT_RATE
    BIT_RATE = value

def set_carrierfreq(value):
    global CARRIER_FREQ
    CARRIER_FREQ = value

TRANSMITTER_PORT = "COM11"
# TRANSMITTER_PORT = "/dev/cu.usbserial-0232D158"
MIC_INDEX = 1 # Mathias, 1 Morten
USE_ESP = False
SAMPLE_RATE = 96000  # this capped by the soundcard, therefore, this is non-changeable

BIT_RATE = 500
CARRIER_FREQ = 11000 
SAMPLES_PER_SYMBOL = int(SAMPLE_RATE / BIT_RATE)
CUT_OFF_FREQ = (CARRIER_FREQ + BIT_RATE) // 2

REP_ESP = 5
# BINARY_BARKER = [1, 1, 1, 0, 0, 1, 0]
# BINARY_BARKER = [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
BINARY_BARKER = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
BINARY_BARKER_CORRECT = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
APPLY_BAKER_PREAMBLE = True
PLOT_PREAMBLE_CORRELATION = False
PATH_TO_WAV_FILE = "Code/dsp/data/testing_and_logging_recording.wav"
# FILE_NAME_DATA_TESTS = "Received_data_for_tests.csv"
# FILE_NAME_DATA_TESTS = "5m_dist_10kHz_unique_payloads.csv"
# FILE_NAME_DATA_TESTS = "1m_distance_payload_barker_similarity_impact.csv"
# FILE_NAME_DATA_TESTS = "1m_distance_carrier_freq_sg_vpp_variable.csv"
# FILE_NAME_DATA_TESTS = "testing_esp.csv"
# FILE_NAME_DATA_TESTS = "spectrography_analysis_esp_sg_without_speaker.csv"
# FILE_NAME_DATA_TESTS = "1m_distance_bitrate_and_carrierfreq_combination.csv"
# FILE_NAME_DATA_TESTS = "computing_freq_impulse.csv"
# FILE_NAME_DATA_TESTS = "Max_bitrate_at_different_distances.csv"
# FILE_NAME_DATA_TESTS = "Varying_payload_sizes.csv"
# FILE_NAME_DATA_TESTS = "Random_payloads.csv"
# FILE_NAME_DATA_TESTS = "Random_payloads_CORRECT_BARKER.csv"
# FILE_NAME_DATA_TESTS = "avg_power_of_rec_signal_purely_for_check_of_interference.csv"
# FILE_NAME_DATA_TESTS = "Average_power_of_received_signal.csv"
# FILE_NAME_DATA_TESTS = "Max_bitrate_at_different_distances_and_best_carrier_freq.csv"
# FILE_NAME_DATA_TESTS = "Conv_encoding_testing.csv"
FILE_NAME_DATA_TESTS = "Signal_generator_simulation_limit_test.csv"

HAMMING_CODING = False
CONVOLUTIONAL_CODING = False

LIST_OF_DATA_BITS = []
if CONVOLUTIONAL_CODING:
    ENCODING = "Convolutional Encoding"
elif HAMMING_CODING:
    ENCODING = "Hamming Encoding"
else:
    ENCODING = "No Encoding"


# IS_ID_SPECIFIED = None


# IS_ID_SPECIFIED = ["31b5a5b2-6a59-4eb3-9c1a-9eeab1593fcb","1cfcea0b-d869-4bdf-84ae-13620c4b6b9e","ba7d0786-a487-4e50-8e8c-d9800f91f564","30db5a79-5d7f-4533-b398-152df0984aec","b03ae9b7-fbdc-4f48-95eb-2c45307b1e7e","11a56a27-b3fe-4389-9c44-1624bdbcfea5","d2d8f4b0-78df-46e7-ae5a-6fe44f8bcc01","aaf7ede2-847e-4057-ad62-f45a604e181c","767a8d5d-d435-4ac4-b895-ae1a59f3a8c6","a9b1787c-762f-4464-897b-fd7f8d557dca","0f136c01-cf75-4755-808a-581ae78cd013","c59f22ef-b60e-49c7-936f-faf4b8eb36c9","c88b4abc-816d-4b76-b779-c210ae7f3a87","c0faac10-5377-48a5-8a09-f8d0872360a4","47b331ed-10dd-4832-8ec2-5e5637339c50","b9051ffc-fe5b-4966-8a23-8ba135e16fa4","7262ca5b-81cc-4fa1-8f16-5851b3e9255e","cf416a73-68c8-4c52-b5a5-69568bc04864","c11f2180-8fe3-4511-a563-ae1cf1eb2634","f98338e3-899c-4ccd-b087-c6a0cc5f0766","d5314ee1-efd6-492c-9de1-911ba23d874c","70147e57-da3d-495d-a2b1-8b66f54195bb","f6c6f290-b0f6-4ea8-b526-8b9ad7fed17c","2c469b32-9573-4a7b-8941-d9cc2e0b623b","2fdb439c-8490-476e-a0f3-f402289d5aa6","b5b32a99-2915-4f3d-850a-8c1211472352","e87b4617-6ad6-441a-8930-02cdcd735f1c","68dbf870-3602-4fdb-a2dc-f3874a3f96d5","fb4462fa-6cce-4158-84e6-160542a3c842","9c0f17e1-ee2a-4719-9fb2-e308bf432058","dc9240a9-caef-4b12-8ed7-0c5e48d407b4","a920d4bb-66aa-4673-af2d-26eebaf1e570","61fa59cd-e202-4e30-9a7c-33cbf46adf3f","619f3c7e-92e8-40e8-951e-6fe215aee34e","985c2189-77d4-4b21-b1e7-38130020da93","d1f9153f-8cc9-4e08-a55e-7194b205d0ff","6913db25-9dbd-4d18-9f49-67978ea4eb3a","32db85ce-1064-4f3d-a518-a5a75d96eab0","8e62f782-a926-42de-8a69-773faad9200c","e5e6f2a7-2b8f-4b1e-b069-71e562b7302c","c5dbfb7a-c2de-497d-910b-eb58af5d0848","5a03106a-d5c0-4f2f-bd28-4c222fe9dcb4","79a01282-2dc6-4c80-a3c7-4c030fc55cd1","0abb0db6-31db-4023-b17e-ac8de4f4e7a1","b0269812-27aa-4ebd-9734-7a1ba82caafb","25e16070-90f5-40e4-a98a-4125a76a31be","f3ac33f8-7abd-4035-9afa-fd6bf014c0ec","8dfd3610-7ece-402a-9144-55e699a7d44d","88f29a33-5fcf-4236-a6eb-97ff4f62375a","7addf629-8ea2-453d-ad6d-156efcbd21fb","1cf4f82f-8ca3-4774-9ad6-65c27ebbb83b","c2dca78d-53e4-4b99-a85f-82b9e9fd6ff3","cd368167-641b-441e-8b1f-5c840dd41e4b","63c6bf5a-e5e5-4c7d-b9e2-dd86525ce012","5157cb0a-7579-4123-aa2c-3dcd3e28ccb7","b9fb2e82-9413-4e9a-98a3-8bae54da6647","4fb9da50-c25f-458e-a992-7aa5e7435725","531d25a3-667c-481c-af11-741cf556a09a","2756c373-ef64-46dc-960e-44e95bf9a6fb","f33b37ea-323f-4811-a6f4-30f563aafe4f","c1334651-3a15-4ac1-88e1-3a08b51e9566","b2e81eb0-4c84-4ad2-8bc0-300904c341dc","da715522-1562-4fea-af22-fe9c474b5514","98bd6f3b-9fb2-40d7-b800-f64d33e35d46","81b8a6df-0a93-42ff-9abc-c366f634643a","32b99f0b-5707-4a0c-b4e9-7199cd4be8f3","e9918dc9-8ad2-4910-90f4-dfc4b3e4666b","ef67e285-bfa1-47f4-9244-7c04a5910ff9","6d1f2081-75ce-48ab-906d-bd0b245231cc","919c4afa-a908-416d-bb68-1e592ea2682b","c5afba08-89a5-4bcc-ad20-1aa719f54aac","577b9132-5039-48a2-af66-80c3a379adf8","f32e4d8c-db63-49b5-9375-623ba08ae874","11e57809-4b9c-400a-b8b9-4efbeca91036","84861956-5e13-455d-a759-0066590b594e","21008346-9fc8-42a4-9176-3e5023a366b1","b755ae5f-2bdc-4637-939a-20a6f6081556","44c027aa-4968-42e1-a8a3-8cec5e645aca","d7e30906-bbec-4012-9ace-3652101d8319","b94c88f4-f3ef-4889-8327-f17ad11b4504","2d2278f4-83ee-4650-beeb-10dfeb9dadef","94d9c78e-b89b-425c-afc8-ae5075758449","2f5e4225-e789-4d55-8981-f3026b714c94","2d75a9ef-6559-4c01-bc40-c9bd61ac2015","2469e3b7-105c-4921-b1d1-9e5c7705b489","67d28fed-8b6d-4c12-a88e-e82caa07c18f","15c9b803-630b-4d8b-aa7e-5bc6bcaea1fb","fff62b7b-89df-4c57-99b4-77af5a684c42","12f0dd6c-6795-4d8c-af5b-d04f8a5953fb","736438db-db82-4a86-961d-ea9c76f17c8e","8d9a3580-56dc-44e0-871f-d140dd20be47","b12b96df-d666-4327-a70c-51ba41d10f9c","8661a5fe-a1d1-42e3-b778-1260c85ea403","f1103fa3-73b3-4af4-938c-12ef0cb0ba79","525dab7d-bb09-45c3-acc5-abf37145fbf8","0949512e-89d4-447c-8ded-3f53c044b288","7b97b510-615e-47e9-afbf-1a94808ab2a1","0808f283-081a-4571-8af7-24f460f716e8","2de2b42a-2a61-44c1-94db-ec236f212ee6","12174ab8-b599-4044-8e14-58bb10a2a397","9953da2c-40bb-4438-98a6-19b747a0eb85","bf935272-bf08-49e5-b391-eb7b4f223821","9beb4978-02f2-4718-995d-771cf5d1e259","4dbe6e9d-d1cc-49c4-b81f-be6ce5b49803","d1a48faa-7cd8-48d9-a502-fd15463620aa","f6a33215-4e02-4ac9-a960-4c8382200fbc","6566c2d9-cc37-4c59-ba88-0ab0e2271f09","f069d470-e93d-4e33-9181-5ad328ead88d","b185bed1-69f5-4fb0-bb5e-74816dff7207","48117953-ff0b-4796-ae63-1aeb2eee09ec","2fe0c886-00ca-4c09-964a-0d4f2b0fbd04","683b891c-1d03-45fe-998f-d2fd3aa02c11","74dbffc0-9644-4247-94a1-873d01dbc1cd","03236721-48b9-4fe4-bf25-c814d3fda814","81cb58dd-2f1f-4ff0-8cc2-4c3feb2ed93b","721c9795-280a-4ca3-8f5d-6115de51570a","cea22ba4-15fe-4be7-8f72-9d47733980fc","12bf0a40-3bc8-4651-91e1-8948530c72af","f350dfbd-6390-4ce4-a5f5-f8c321a9837a","72d74cb3-f9f2-443c-95f7-1e10c830f514","ee85b314-b32c-42f9-acfe-ce4ceb60e881","565270ba-9205-4ec7-9c0a-15bb05218f0f","80c31146-4492-437e-be40-9981f760ebac","25b3983f-b044-4d83-b4a6-f2befaaf4602","8ffbd26e-a25d-4e3c-bf7c-eb618a9f90d2","60f8d1c4-a945-481e-a5b3-1ac2337ba14f","199c0da1-9a8e-43a2-9bb6-fe8c583916d6","804d49ea-582c-4333-a675-a3b6634c3722","937b35e4-890e-4e4e-adfd-b82508b0bd4e","a21c4881-5aef-42d9-a394-3dcb12da7be5","8e8ddd38-de6a-4737-84fc-9c79865488b1","9e064d91-5f07-4847-843c-210144af47d1","fc67c12c-d10b-435c-a2a6-1b4a77218955","e3121a1c-01d6-4b23-a5a8-da51d787ea43","314a8ecc-dea7-4a91-b345-6ba64e79bcca","86dcaeba-097e-47aa-b4b1-6dcbe4c5b0e2","a2d4c05e-27c2-40c1-9206-0a44db29fccc","badccf8d-8065-4c65-9ca0-17a55483d958","bdbc4264-2ac3-4812-8c00-79a32c8558d0","e2ca7042-2379-476c-9365-252ce59b1473","0c1940b2-77ea-43c4-9dd5-f7f93f63b5fa","563cf08a-4ffb-43b9-b559-ffe57fb60920","c6540aff-fa08-4c6d-aa8f-c2ccff5bf671","a042d344-3dfc-4774-8e93-2db86c0969dc","dba81387-7b8d-44c3-9c56-46ef12f56400","863537da-6edf-40c1-8610-93ffd013d505","cf0ec5d8-764a-427a-bb22-acd5f82d5cd7","97ebcfff-a9e4-4984-b684-8a4e89c82d9e","37c47d1a-12ce-44f6-933b-7aede45b436f","08b3bfef-a60f-4ca2-be63-7c73fa473b88","cadb3816-4e76-42c1-a4ec-32357bf34fb4","5cfcca3c-ef4a-49d1-b26b-237ca805c321","566193db-8a44-43ce-bebd-5c43a08ac9c3","d21c082d-1421-402d-8a88-0dfc1ad6fc6f","5e31e182-3c61-449a-80f5-01463375cea6","3c9e67ff-8395-478a-a576-c399f5d9da7e","0b7caa9f-85da-4407-b4b9-0fede621e40d"]


# IS_ID_SPECIFIED = ["948ebdd5-0edd-47e3-ae00-18c75a484194"] # esp32 attenuated 4000 bits
# IS_ID_SPECIFIED = ["7c3036d6-654a-4b77-8fe2-2cfc6697f732"] # ESP straight into computer
# IS_ID_SPECIFIED = ["190209f4-1a18-48f7-a33f-7058cc78ac3b"] # ESP straight into computer 4000 bits attenuaed
# IS_ID_SPECIFIED = ["9d552825-968b-46b4-8c76-41f58e663ccc"]
IS_ID_SPECIFIED = ["e7014386-5e79-4a51-b86d-278678b9e8c3"]
# IS_ID_SPECIFIED = ["8a58d36b-96f7-424e-b9c8-6cad8323b037"] # bitrate 1000, cf 10000, esp 


if __name__ == "__main__":
    # import scipy.signal as signal
    
    # print(np.arange(50 // 8, 1050 // 8, 50 // 8))
    messages = "-gF:]#N9tjH;6neUxbNRV^ydrmz%m!r}"
    bin_msg = string_to_bin_array(messages)
    autocorrelate = signal.correlate(bin_msg, BINARY_BARKER, mode='same')

    print(autocorrelate)
