#include "transmitter.h"

void transmit(char* message)
{
    dac_output_enable(DAC_CHAN_0); //TODO what is this used for??
    
    // Convert string to bits

    int* message_bits = NULL;
    int message_length = 0;
    string_to_bits(message, &message_bits, &message_length);
    
    const int total_samples = message_length * SAMPLES_PER_SYMBOL;
    
    // Allocate memory
    int* square_wave = malloc(total_samples * sizeof(int));
    float* time_array = malloc(total_samples * sizeof(float));
    int* carrier_wave = malloc(total_samples * sizeof(int));
    int* modulated_wave = malloc(total_samples * sizeof(int));
    
    const float time_increment = 1.0f / SAMPLE_RATE;
    
    // Generate waves
    create_square_wave(square_wave, message_bits, message_length);
    create_time_array(time_array, total_samples, time_increment);
    create_carrier_wave(time_array, carrier_wave, total_samples);
    create_modulated_wave(carrier_wave, square_wave, modulated_wave, total_samples);
    
    // Transmit
    send_wave(modulated_wave, total_samples);
    
    // Cleanup
    free(message_bits);
    free(square_wave);
    free(time_array);
    free(carrier_wave);
    free(modulated_wave);
}
void string_to_bits(const char* str, int** bits, int* length)
{
    const int str_len = strlen(str);
    *length = str_len * 8;
    *bits = malloc(*length * sizeof(int));
    
    for(int i = 0; i < str_len; i++) {
        char c = str[i];
        for(int b = 7; b >= 0; b--) {
            (*bits)[i*8 + (7-b)] = (c >> b) & 0x01;
        }
    }
}
void create_square_wave(int* square_wave, const int* message, int message_length)
{
    for(int i = 0; i < message_length; i++) {
        for(int j = 0; j < SAMPLES_PER_SYMBOL; j++) {
            square_wave[i*SAMPLES_PER_SYMBOL + j] = message[i];
        }
    }
}
void create_time_array(float* time_array, int total_samples, float time_increment)
{
    for(int i = 0; i < total_samples; i++) {
        time_array[i] = i * time_increment;
    }
}
void create_carrier_wave(const float* time_array, int* carrier_wave, int total_samples)
{
    for(int i = 0; i < total_samples; i++) {
        float value = round(128 * sin(2 * M_PI * CARRIER_FREQ * time_array[i])) + 128; // 1kHz carrier
        carrier_wave[i] = (int)fmin(fmax(round(value) + 128, 0), 255);
    }
}
void create_modulated_wave(const int* carrier_wave, const int* square_wave, int* modulated_wave, int total_samples)
{
    for(int i = 0; i < total_samples; i++) {
        modulated_wave[i] = (carrier_wave[i] * square_wave[i]);
    }
}
void send_wave(const int* modulated_wave, int total_samples)
{   
    while true(1):
        for(int i = 0; i < total_samples; i++) {
            dac_output_voltage(DAC_CHAN_0, modulated_wave[i]);
            vTaskDelay(pdMS_TO_TICKS(1000/SAMPLE_RATE));
            printf("Modulated wave: %d\n", modulated_wave[i]);
        }
}