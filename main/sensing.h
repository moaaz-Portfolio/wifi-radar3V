#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

typedef struct {
    bool running;
    bool ready;
    bool motion;
    bool presence;
    float jitter;
    float wander;
    int train_status;
    bool train_valid;
    bool train_running;
    bool train_complete;
    uint16_t train_elapsed_s;
    uint16_t train_remaining_s;
    float trained_wander;
    float trained_jitter;
} sensing_snapshot_t;

esp_err_t sensing_start(void);
esp_err_t sender_start(void);
void sensing_apply_settings(void);
void sensing_get(sensing_snapshot_t *out);

esp_err_t sensing_train_start(void);
esp_err_t sensing_train_stop(float *wander_th, float *jitter_th);
esp_err_t sensing_train_remove(void);
esp_err_t sensing_reset_baseline(void);
