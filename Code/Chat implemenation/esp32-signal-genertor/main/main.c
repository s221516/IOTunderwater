#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_err.h"
#include "driver/dac.h"
#include "esp_bt.h"
#include "esp_spp_api.h"
#include "esp_gap_bt_api.h"
#include "esp_bt_main.h"
#include "transmitter.h"

#define SPP_SERVER_NAME "ESP32_Bluetooth"
#define TAG "BT_SPP"

static uint32_t client_handle = 0;

void bt_spp_callback(esp_spp_cb_event_t event, esp_spp_cb_param_t *param) {
    switch (event) {
        case ESP_SPP_SRV_OPEN_EVT:
            ESP_LOGI(TAG, "Client Connected!");
            client_handle = param->srv_open.handle;  // Store client connection handle
            break;

        case ESP_SPP_DATA_IND_EVT:
            char* message = (char*)param->data_ind.data;
            ESP_LOGI(TAG, "Received: %.*s", param->data_ind.len, message);
            transmit(message);
            break;

        case ESP_SPP_CLOSE_EVT:
            ESP_LOGI(TAG, "Client Disconnected!");
            client_handle = 0;  // Reset connection handle
            break;
    }
}

void init_bluetooth() {
    esp_bt_controller_mem_release(ESP_BT_MODE_BLE);
    esp_bt_controller_config_t bt_cfg = BT_CONTROLLER_INIT_CONFIG_DEFAULT();
    esp_bt_controller_init(&bt_cfg);
    esp_bt_controller_enable(ESP_BT_MODE_CLASSIC_BT);
    esp_bluedroid_init();
    esp_bluedroid_enable();
    esp_spp_register_callback(bt_spp_callback);
    esp_spp_init(ESP_SPP_MODE_CB);
    esp_spp_start_srv(ESP_SPP_SEC_AUTHENTICATE, ESP_SPP_ROLE_SLAVE, 0, SPP_SERVER_NAME);
    ESP_LOGI(TAG, "Bluetooth SPP Server Initialized!");
}

void app_main(void) {
    init_bluetooth();
    xTaskCreate(dac_output_task, "DAC_Task", 2048, NULL, 5, NULL);
}
