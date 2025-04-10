#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_err.h"
#include "driver/dac.h"
#include "rom/ets_sys.h"
#include <math.h>
#include "esp_task_wdt.h"
#include <stdlib.h>
#include "esp_timer.h"
#include "driver/uart.h"


#define PI 3.14159265

void send_wave_DAC(int* wave, int samples, int sample_rate_dac) {
    for (int i = 0; i < samples; i++) {
        dac_output_voltage(DAC_CHAN_0, wave[i]);
        //ets_delay_us(1000000 / sample_rate_dac); //sample rate + instructioner delay
    }
}

void string_to_bits(const char* str, int** bits, int* length) {
    int str_len = strlen(str);
    *length = str_len * 8;
    *bits = malloc(*length * sizeof(int));

    for (int i = 0; i < str_len; i++) {
        for (int b = 7; b >= 0; b--) {
            (*bits)[i * 8 + (7 - b)] = (str[i] >> b) & 0x01;
        }
    }
}

void create_carrier_symbol(int* buffer, int samples_per_symbol, int carrier_freq, int sample_rate) {
    
    for (int i = 0; i < samples_per_symbol; i++) {

        //find time
        float t = (float)i / sample_rate;

        //find sin value, and scale it to 0-255
        float s = sin(2 * PI * carrier_freq * t);
        int val = (int)(s * 127.0f + 128.0f);
        if (val < 0) val = 0;
        if (val > 255) val = 255;

        //add wave value to the buffer
        buffer[i] = val;
    }
}

void create_silence_symbol(int* buffer, int samples_per_symbol) {
    for (int i = 0; i < samples_per_symbol; i++) {
        buffer[i] = 128;  // Midpoint (zero amplitude)
    }
}

void append_barker13(int** bits, int* length) {
    // Barker 13 sequence as bits: +1 -> 1, -1 -> 0
    int barker13[] = {1,1,1,1,1,0,0,1,1,0,1,0,1};
    int barker_len = sizeof(barker13) / sizeof(barker13[0]);

    int new_length = *length + barker_len;
    *bits = realloc(*bits, new_length * sizeof(int));
    if (*bits == NULL) {
        ESP_LOGE("append_barker13", "Failed to realloc bits buffer.");
        return;
    }

    for (int i = 0; i < barker_len; i++) {
        (*bits)[*length + i] = barker13[i];
    }

    *length = new_length;
}


void app_main() {
    printf("HELLO LIL N...\n");
    esp_task_wdt_deinit();  // Turn off watchdog for testing
    dac_output_enable(DAC_CHAN_0);

    const char* message = "Hello there";
    int* message_bits = NULL;
    int message_length = 0;


    string_to_bits(message, &message_bits, &message_length);
    append_barker13(&message_bits, &message_length);

    //print the bit array
    printf("Message bits: ");
    for (int i = 0; i < message_length; i++) {
        printf("%d", message_bits[i]);
    }

    int sampleRates[] = {
        110000
    }; 
    int sampleRatesLength = sizeof(sampleRates) / sizeof(sampleRates[0]);
    
    int bitRates[] = {100};
    int bitRatesLength = sizeof(bitRates) / sizeof(bitRates[0]);

    int carrierFrequencies[] = {2000};
    int carrierFrequenciesLength = sizeof(carrierFrequencies) / sizeof(carrierFrequencies[0]);

    //sweep over the above

    
    for (int i = 0; i < sampleRatesLength; i++) {
        for (int j = 0; j < bitRatesLength; j++) {
            for (int k = 0; k < carrierFrequenciesLength; k++) {
                
                //print whats about to be send
                printf("Sample Rate: %d, Bit Rate: %d, Carrier Frequency: %d\n", sampleRates[i], bitRates[j], carrierFrequencies[k]);
                vTaskDelay(1000 / portTICK_PERIOD_MS);
                

                int samples_per_symbol = sampleRates[i] / bitRates[j];

                // Allocate just two waveforms: one for bit '1', one for bit '0'
                int* symbol_one = malloc(samples_per_symbol * sizeof(int));
                int* symbol_zero = malloc(samples_per_symbol * sizeof(int));

                if (!symbol_one || !symbol_zero) {
                    ESP_LOGE("main", "Failed to allocate symbol buffers.");
                    return;
                }

                create_carrier_symbol(symbol_one, samples_per_symbol, carrierFrequencies[k], sampleRates[i]);
                create_silence_symbol(symbol_zero, samples_per_symbol);

                //send message many times
                for (int i1 = 0; i1 < 30; i1++) {
                    
                    // int64_t start_time = esp_timer_get_time(); // microseconds
                    // send_wave_DAC(symbol_one, samples_per_symbol, sampleRates[i]);
                    // int64_t end_time = esp_timer_get_time();

                    // double duration_sec = (end_time - start_time) / 1000000.0;
                
                    // printf("%lld,%d,%d\n", end_time - start_time, bitRates[j], sampleRates[i]);


                    for (int j1 = 0; j1 < message_length; j1++) {
                        if (message_bits[j1]) {
                            send_wave_DAC(symbol_one, samples_per_symbol, sampleRates[i]);
                        } else {
                            send_wave_DAC(symbol_zero, samples_per_symbol, sampleRates[i]);
                        }
                    }
                }

                printf("Done.\n");
                free(symbol_one);
                free(symbol_zero);
            }
        }
    } 
    
    
    free(message_bits);
}
