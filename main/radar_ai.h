#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

typedef struct {
    bool model_ok;
    bool ready;
    int label;
    int best;
    float confidence;
    float probs[4];
    uint32_t infer_ms;
    uint32_t count;
    const char *error;
} radar_ai_snapshot_t;

esp_err_t radar_ai_start(void);
void radar_ai_get(radar_ai_snapshot_t *out);
