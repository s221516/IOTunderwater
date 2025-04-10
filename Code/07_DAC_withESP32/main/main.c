#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "esp_log.h"
#include "driver/dac.h"
#include "rom/ets_sys.h"
#include <math.h>
#include "esp_timer.h"
#include "driver/uart.h"
#include "esp_system.h"
#include "esp_task_wdt.h"

#define UART_NUM UART_NUM_0
#define BUF_SIZE (1024)
#define MAX_LINE_LENGTH 128
#define PI 3.14159265

int bit_rate = 100;
int carrier_freq = 6000;
char message[100] = "Hello World!"; 
int samples_per_symbol = 0;
float sample_rate = 0;
int repetitions = 10;

void transmit_symbol(int* wave) {
    for (int i = 0; i < samples_per_symbol; i++) {
        dac_output_voltage(DAC_CHAN_0, wave[i]);
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

void create_carrier_symbol(int* buffer) {
    for (int i = 0; i < samples_per_symbol; i++) {
        float t = (float)i / sample_rate;
        float s = sin(2 * PI * carrier_freq * t);
        int val = (int)(s * 127.0f + 128.0f);
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        buffer[i] = val;
    }
}

void create_silence_symbol(int* buffer) {
    for (int i = 0; i < samples_per_symbol; i++) {
        buffer[i] = 128;  // Midpoint (zero amplitude)
    }
}

void prepend_barker13(int** bits, int* length) {
    int barker13[] = {1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1};
    int barker_len = sizeof(barker13) / sizeof(barker13[0]);

    int new_length = *length + barker_len;
    *bits = realloc(*bits, new_length * sizeof(int));
    if (*bits == NULL) {
        ESP_LOGE("prepend_barker13", "Failed to realloc bits buffer.");
        return;
    }

    for (int i = *length - 1; i >= 0; i--) {
        (*bits)[i + barker_len] = (*bits)[i];
    }

    for (int i = 0; i < barker_len; i++) {
        (*bits)[i] = barker13[i];
    }

    *length = new_length;
}

float measureSampleRateOfDAC() {
    const int num_samples = 100000;
    int64_t start_us = esp_timer_get_time();

    for (int i = 0; i < num_samples; i++) {
        uint8_t value = i % 255;
        dac_output_voltage(DAC_CHAN_0, value);
    }

    int64_t end_us = esp_timer_get_time();
    float elapsed_seconds = (end_us - start_us) / 1e6;
    float sample_rate = num_samples / elapsed_seconds;

    return sample_rate;
}

void send_signal() {
    int* message_bits = NULL;
    int message_length = 0;

    string_to_bits(message, &message_bits, &message_length);
    prepend_barker13(&message_bits, &message_length);

    samples_per_symbol = (int) roundf(sample_rate / bit_rate);
    int* symbol_one = malloc(samples_per_symbol * sizeof(int));
    int* symbol_zero = malloc(samples_per_symbol * sizeof(int));
    
    if (!symbol_one || !symbol_zero) {
        ESP_LOGE("main", "Failed to allocate symbol buffers.");
        return;
    }

    create_carrier_symbol(symbol_one);
    create_silence_symbol(symbol_zero);
    
    float sampleRate_after_allocations = measureSampleRateOfDAC(); 

    printf("\n");
    printf("Sample rate before allocations: %.2f\n", sample_rate);
    printf("Sample rate after allocations: %.2f\n", sampleRate_after_allocations);
    printf("Samples per symbol: %d\n", samples_per_symbol);
    printf("Carrier frequency: %d\n", carrier_freq);
    printf("Bit rate: %d\n", bit_rate);
    printf("Message length: %d\n", message_length);
    printf("Message: %s\n", message);
    printf("Message bits: ");
    
    for (int i = 0; i < message_length; i++) {
        printf("%d", message_bits[i]);
    }
    printf("\n");
    printf("\n");

    for (int i1 = 0; i1 < repetitions; i1++) {
        for (int j1 = 0; j1 < message_length; j1++) {
            if (message_bits[j1]) {
                transmit_symbol(symbol_one);
            } else {
                transmit_symbol(symbol_zero);
            }
        }
    }
    
    printf("Done.\n");
    free(symbol_one);
    free(symbol_zero);
    free(message_bits);
}

void process_input(char *input) {
    int value;
    char valueStr[100];
    

    printf("\n");

    if (sscanf(input, "FREQ %d", &value) == 1) {
        carrier_freq = value;
    
    } else if (sscanf(input, "BITRATE %d", &value) == 1) {
        bit_rate = value;

    } else if (sscanf(input, "REP %d", &value) == 1) {
        repetitions = value;

    } else if (strcmp(input, "INFO") == 0 || strcmp(input, "") == 0) {
        //print info on settings
        printf("Current settings:\n");
        printf("Carrier frequency: %d\n", carrier_freq);
        printf("Bit rate: %d\n", bit_rate);
        printf("Wave Repetitions: %d\n", repetitions);
    
    } else if (strcmp(input, "HELP") == 0 || strcmp(input, "") == 0) {
        //List of commands
        printf("Commands:\n");
        printf("FREQ <value>    - Set carrier frequency         (default: 6000)\n");
        printf("BITRATE <value> - Set bit rate                  (default: 100)\n");
        printf("REP <value>     - Set number of repetitions     (default: 10)\n");
        printf("INFO            - Show current settings\n");
        printf("anything else to send a message (finish with enter)\n");

    
    }
     else {

        if (sscanf(input, "%99s", valueStr) == 1) {
            strncpy(message, valueStr, sizeof(message) - 1);    
            message[sizeof(message) - 1] = '\0';  // Ensure null termination
            printf("Sending message: %s\n", message);
            send_signal();
        } else {
            printf("Failed to parse message from input: %s\n", input);
        }
    }
}

void init_uart() {
    uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    };
    uart_driver_install(UART_NUM, BUF_SIZE * 2, 0, 0, NULL, 0);
    uart_param_config(UART_NUM, &uart_config);
}

void app_main() {

    dac_output_enable(DAC_CHAN_0);
    //deinit watchdog
    esp_task_wdt_deinit();
    sample_rate = measureSampleRateOfDAC();
    init_uart();

    char line_buffer[MAX_LINE_LENGTH];
    int idx = 0;

    //type HELP to see the commands
    printf("\n MORTIII type HELP to see commands\n");

    while (1) {
        uint8_t byte;
        int len = uart_read_bytes(UART_NUM, &byte, 1, 20 / portTICK_PERIOD_MS);
        if (len > 0) {
            if (byte == '\r' || byte == '\n') {
                uart_write_bytes(UART_NUM, "\r\n", 2); 
                line_buffer[idx] = '\0';
                if (idx > 0) {
                    process_input(line_buffer);
                }
                idx = 0;
            } else if (byte == 0x08 || byte == 0x7F) {
                if (idx > 0) {
                    idx--;
                    const char *bs_seq = "\b \b";
                    uart_write_bytes(UART_NUM, bs_seq, strlen(bs_seq));
                }
            } else if (idx < MAX_LINE_LENGTH - 1) {
                line_buffer[idx++] = byte;
                uart_write_bytes(UART_NUM, (const char *)&byte, 1); 
            }
        }
        vTaskDelay(10 / portTICK_PERIOD_MS);
    }
}
