#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

typedef struct {
    bool enabled;
    bool sensor_ok;
    bool distance_valid;
    bool present;
    bool buzzer_active;
    bool crossing;
    bool sensor_fault;
    bool baseline_valid;
    uint16_t baseline_cm;
    uint16_t distance_cm;
} door_sensor_snapshot_t;

esp_err_t door_sensor_start(void);
void door_sensor_get(door_sensor_snapshot_t *out);
esp_err_t door_sensor_reset_baseline(void);
