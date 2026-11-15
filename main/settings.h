#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"
#include "cJSON.h"

typedef struct {

    char wifi_ssid[33];
    char wifi_pass[65];
    char hostname[32];
    char room_name[48];

    uint8_t role;
    char stream_host[40];
    uint16_t stream_port;
    char pair_mac[18];
    uint8_t source;

    float sensitivity;
    float presence_sensitivity;
    float jitter_min;
    uint32_t filter_ms;
    uint8_t ping_hz;
    uint16_t hold_s;

    uint8_t led_type;
    int8_t led_gpio;
    uint8_t led_brightness;
    char led_color[8];

    int8_t out_gpio;
    bool out_active_high;
    uint16_t out_hold_s;

    bool door_enabled;
    bool buzzer_enabled;
    uint16_t door_threshold_cm;
    uint16_t door_baseline_cm;
    bool door_baseline_valid;

    char lang[3];
    char ui_pass[33];

    bool mqtt_en;
    char mqtt_host[64];
    uint16_t mqtt_port;
    char mqtt_user[33];
    char mqtt_pass[65];
    char mqtt_prefix[32];
    bool mqtt_ha;

    char webhook_url[128];
} settings_t;

settings_t *settings_get(void);
void settings_defaults(settings_t *s);
esp_err_t settings_load(void);
esp_err_t settings_save(void);
esp_err_t settings_wifi_reset(void);
esp_err_t settings_reset_all(void);
bool settings_has_wifi(void);

cJSON *settings_to_json(const settings_t *s, bool include_secrets);
void settings_apply_json(settings_t *s, const cJSON *j, bool trusted);

bool settings_parse_mac(const char *str, uint8_t out[6]);
