#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_err.h"
#include "driver/dac.h"
#include "transmitter.h"



void app_main(void) {
    printf("v3...\n");

    for(int i = 0; i < 100; i++) {
        transmit("A");
    }
}
