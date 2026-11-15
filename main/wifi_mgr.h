#pragma once
#include <stdbool.h>
#include "esp_err.h"

typedef enum { NETMODE_STA = 0, NETMODE_SETUP = 1 } netmode_t;

netmode_t wifi_mgr_start(void);
netmode_t wifi_mgr_mode(void);
const char *wifi_mgr_ip(void);
bool wifi_mgr_connected(void);

const char *wifi_mgr_scan(bool refresh);

void wifi_mgr_touch(void);
