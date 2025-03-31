#ifndef TRANSMITTER_H
#define TRANSMITTER_H

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/dac.h"
#include "esp_system.h"
#include "esp_mac.h"
#include <rom/ets_sys.h>
#include "esp_task_wdt.h"



// Fixed macros
// #define barker_code [1,1,1,1,1,0,0,1,1,0,1,0,1]
#define BIT_RATE 100
#define SAMPLE_RATE 10000
#define SAMPLES_PER_SYMBOL (SAMPLE_RATE / BIT_RATE)
#define CARRIER_FREQ 4000
    
// Function declarations
void string_to_bits(const char* str, int** bits, int* length);
void create_square_wave(int* square_wave, const int* message, int message_length);
void create_time_array(float* time_array, int total_samples, float time_increment);
void create_carrier_wave(const float* time_array, int* carrier_wave, int total_samples);
void create_modulated_wave(const int* carrier_wave, const int* square_wave, 
                           int* modulated_wave, int total_samples);
void send_wave(const int* modulated_wave, int total_samples);
void transmit(char* message);

#endif // TRANSMITTER_H
