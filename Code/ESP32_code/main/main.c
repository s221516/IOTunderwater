#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "esp_log.h"
#include "rom/ets_sys.h"
#include <math.h>
#include "esp_timer.h"
#include "driver/uart.h"
#include "esp_system.h"
#include "esp_task_wdt.h"
#include "driver/dac_oneshot.h"

#define UART_NUM UART_NUM_0
#define BUF_SIZE (1024)
#define MAX_LINE_LENGTH 1000
#define PI 3.14159265
#define INPUT_SIZE 500

static dac_oneshot_handle_t dac_handle;

int bit_rate = 100;
int carrier_freq = 6000;
char message[INPUT_SIZE] = "Hello World!";
int samples_per_symbol = 0;
float sample_rate = 0;
int repetitions = 1;

void transmit_symbol(int* wave) {

    for (int i = 0; i < samples_per_symbol; i++) {
        dac_oneshot_output_voltage(dac_handle, wave[i]);
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

void create_scaled_carrier_symbol(int* buffer, float amplitude_scale) {
    for (int i = 0; i < samples_per_symbol; i++) {
        float t = (float)i / sample_rate;
        float s = sin(2 * PI * carrier_freq * t) * amplitude_scale;
        float scaled = s * 127.0f + 128.0f;
        int val = (int)(scaled);
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        buffer[i] = val;
    }
}


float measureSampleRateOfDAC() {
    const int num_samples = 1000000;
    int64_t start_us = esp_timer_get_time();
    for (int i = 0; i < num_samples; i++) {
        uint8_t value = i % 255;
        dac_oneshot_output_voltage(dac_handle, value);
    }
    int64_t end_us = esp_timer_get_time();
    float elapsed_seconds = (end_us - start_us) / 1e6;
    return num_samples / elapsed_seconds;
}

void send_signal() {
    int* message_bits = NULL;
    int message_length = 0;
    string_to_bits(message, &message_bits, &message_length);
    prepend_barker13(&message_bits, &message_length);
    samples_per_symbol = roundf(sample_rate / bit_rate);
    int* symbol_one = malloc(samples_per_symbol * sizeof(int));
    int* symbol_zero = malloc(samples_per_symbol * sizeof(int));

    printf("%d \n", message_length);
    
    if (!symbol_one || !symbol_zero) {
        ESP_LOGE("main", "Failed to allocate symbol buffers.");
        return;
    }

    
    float samples_per_carrier = sample_rate / carrier_freq;
    if (samples_per_symbol < samples_per_carrier) {
        ESP_LOGE("main", "Bitrate too high or carrier frequency too low: 1-symbol cannot fit full carrier cycle.");
        const char *error_msg = "ERROR: Symbol duration too short for full carrier wave. Try lowering bitrate or increasing frequency.\r\n";
        uart_write_bytes(UART_NUM, error_msg, strlen(error_msg));
        return;
    }

    create_scaled_carrier_symbol(symbol_one, 1.0f);  // Bit 1 → full amplitude
    create_scaled_carrier_symbol(symbol_zero, 0.1f); // Bit 0 → low amplitude

    // UART print of settings
    char settings_message[256];
    snprintf(settings_message, sizeof(settings_message),
             "Carrier Frequency: %d Hz, Bit Rate: %d bps, Sample Rate: %.2f Hz, Samples per Symbol: %d\r\n",
             carrier_freq, bit_rate, sample_rate, samples_per_symbol);
    uart_write_bytes(UART_NUM, settings_message, strlen(settings_message));

    for (int i = 0; i < message_length; i++) {
        printf("%d", message_bits[i]);
    }

    for (int i1 = 0; i1 < repetitions; i1++) {
        for (int j1 = 0; j1 < message_length; j1++) {
            transmit_symbol(message_bits[j1] ? symbol_one : symbol_zero);
        }
    }

    const char *done_message = "Signal transmission completed.\r\n";
    uart_write_bytes(UART_NUM, done_message, strlen(done_message));
    free(symbol_one);
    free(symbol_zero);
    free(message_bits);
}

void process_input(char *input) {
    int value;
    
    if (sscanf(input, "FREQ %d", &value) == 1) {
        sample_rate = measureSampleRateOfDAC();
        carrier_freq = value;

    } else if (sscanf(input, "BITRATE %d", &value) == 1) {
        sample_rate = measureSampleRateOfDAC();
        bit_rate = value;

    } else if (sscanf(input, "REP %d", &value) == 1) {
        sample_rate = measureSampleRateOfDAC();
        repetitions = value;

    } else {
        strncpy(message, input, sizeof(message) - 1);
        message[sizeof(message) - 1] = '\0';  // Ensure null-termination
        send_signal();
    }

    const char *ok_message = "OK\r\n";
    uart_write_bytes(UART_NUM, ok_message, strlen(ok_message));
}

void init_uart() {
    uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    };
    uart_driver_install(UART_NUM, BUF_SIZE * 2, BUF_SIZE * 2, 0, NULL, 0);
    uart_param_config(UART_NUM, &uart_config);
}

void app_main() {
    init_uart();
    
    dac_oneshot_config_t dac_cfg = {
        .chan_id = DAC_CHAN_0,
    };
    ESP_ERROR_CHECK(dac_oneshot_new_channel(&dac_cfg, &dac_handle));
    
    esp_task_wdt_deinit();

    char line_buffer[MAX_LINE_LENGTH];
    int idx = 0;


    sample_rate = measureSampleRateOfDAC(); // Measure DAC sample rate
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
            } else if (idx < MAX_LINE_LENGTH - 1) {
                line_buffer[idx++] = byte;
            }
        }
        vTaskDelay(10 / portTICK_PERIOD_MS);
    }
}
