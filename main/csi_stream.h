#pragma once
#include <stdint.h>
#include "esp_err.h"

esp_err_t csi_stream_start(void);
uint32_t csi_stream_sent(void);
