#include "sensing.h"
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_now.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "esp_wifi_sensing.h"

#include "settings.h"
#include "outputs.h"
#include "notify.h"
#include "door_sensor.h"

static const char *TAG = "sensing";

static esp_wifi_sensing_fsm_handle_t s_fsm = NULL;
static uint8_t s_ap_mac[6];
static uint8_t s_pair_mac[6];
static bool s_has_pair = false;

static volatile bool s_ap_active = false, s_pair_active = false;
static bool s_motion = false, s_presence = false;
static int64_t s_last_motion_us = INT64_MIN / 2;
static sensing_snapshot_t s_snap;
static volatile bool s_train_running;
static volatile bool s_train_complete;
static int64_t s_train_started_us;
#define ROOM_CALIBRATION_SECONDS 600

static const uint8_t BROADCAST[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

static void on_event(esp_wifi_sensing_fsm_handle_t handle, const uint8_t peer_mac[6],
                     esp_wifi_sensing_fsm_event_t event, uint32_t data, void *usr)
{
    (void)handle; (void)data; (void)usr;
    bool active = (event == ESP_WIFI_SENSING_FSM_EVENT_ACTIVE);
    if (memcmp(peer_mac, s_ap_mac, 6) == 0) s_ap_active = active;
    else if (s_has_pair && memcmp(peer_mac, s_pair_mac, 6) == 0) s_pair_active = active;
}

static void espnow_rx_cb(const esp_now_recv_info_t *info, const uint8_t *d, int len)
{
    (void)info; (void)d; (void)len;
}

static const uint8_t *primary_mac(void)
{
    return (settings_get()->source == 1 && s_has_pair) ? s_pair_mac : s_ap_mac;
}

static void logic_task(void *arg)
{
    for (;;) {
        settings_t *c = settings_get();
        int64_t now = esp_timer_get_time();

        bool raw = (c->source == 0) ? s_ap_active
                 : (c->source == 1) ? s_pair_active
                 : (s_ap_active || s_pair_active);
        if (raw) s_last_motion_us = now;
        bool shown = raw || (now - s_last_motion_us) < (int64_t)c->hold_s * 1000000LL;

        if (shown != s_motion) {
            s_motion = shown;
            outputs_set_motion(shown);
            notify_post(NOTIFY_MOTION, shown);
        }

        esp_wifi_sensing_fsm_channel_diag_t diag = {0};
        if (s_fsm && esp_wifi_sensing_fsm_get_channel_diag(s_fsm, primary_mac(), &diag) == ESP_OK) {
            s_snap.jitter = diag.jitter_value;
            s_snap.wander = diag.wander_value;
            s_snap.ready = (diag.init_stage == ESP_WIFI_SENSING_FSM_INIT_STAGE_STABLE);
            s_snap.train_status = (int)diag.train_status;
            s_snap.train_valid = diag.train_thresholds_valid;
            bool pres = diag.presence_someone_status;
            if (pres != s_presence) {
                s_presence = pres;
                notify_post(NOTIFY_PRESENCE, pres);
            }
        }
        s_snap.motion = s_motion;
        s_snap.presence = s_presence;
        s_snap.train_running = s_train_running;
        s_snap.train_complete = s_train_complete;
        if (s_train_running) {
            uint32_t elapsed = (uint32_t)((now - s_train_started_us) / 1000000LL);
            if (elapsed >= ROOM_CALIBRATION_SECONDS) {
                float w = 0.0f, j = 0.0f;
                s_train_running = false;
                esp_err_t train_err = esp_wifi_sensing_fsm_train_stop(s_fsm, primary_mac(), &w, &j);
                s_train_complete = (train_err == ESP_OK);
                if (train_err == ESP_OK) {
                    s_snap.trained_wander = w;
                    s_snap.trained_jitter = j;
                    ESP_LOGI(TAG, "10-minute room calibration completed (wander=%.3f jitter=%.3f)", w, j);
                } else {
                    ESP_LOGE(TAG, "automatic calibration stop failed: %s", esp_err_to_name(train_err));
                }
            }
        }
        if (s_train_running) {
            uint32_t elapsed = (uint32_t)((now - s_train_started_us) / 1000000LL);
            s_snap.train_elapsed_s = elapsed > ROOM_CALIBRATION_SECONDS ? ROOM_CALIBRATION_SECONDS : (uint16_t)elapsed;
            s_snap.train_remaining_s = (uint16_t)(ROOM_CALIBRATION_SECONDS - s_snap.train_elapsed_s);
        } else {
            s_snap.train_remaining_s = 0;
        }

        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

esp_err_t sensing_start(void)
{
    settings_t *c = settings_get();

    wifi_ap_record_t ap = {0};
    esp_err_t e = esp_wifi_sta_get_ap_info(&ap);
    if (e != ESP_OK) { ESP_LOGE(TAG, "not connected to an AP"); return e; }
    memcpy(s_ap_mac, ap.bssid, 6);
    s_has_pair = settings_parse_mac(c->pair_mac, s_pair_mac);

    if (s_has_pair) {
        esp_err_t ne = esp_now_init();
        if (ne == ESP_OK || ne == ESP_ERR_ESPNOW_EXIST) {
            esp_now_register_recv_cb(espnow_rx_cb);
            if (!esp_now_is_peer_exist(BROADCAST)) {
                esp_now_peer_info_t p = {0};
                memcpy(p.peer_addr, BROADCAST, 6);
                p.channel = 0;
                p.ifidx = WIFI_IF_STA;
                esp_now_add_peer(&p);
            }
        }
    }

    esp_wifi_sensing_fsm_config_t cfg = DEFAULT_ESP_WIFI_SENSING_FSM_CONFIG();
    cfg.max_channel_num = s_has_pair ? 2 : 1;
    cfg.default_channel_config.sensitivity = c->sensitivity;
    cfg.default_channel_config.presence_sensitivity = c->presence_sensitivity;
    cfg.default_channel_config.active_jitter_min = c->jitter_min;
    cfg.default_channel_config.active_filter_ms = c->filter_ms;
    cfg.ping_frequency_hz = c->ping_hz;

    ESP_ERROR_CHECK(esp_wifi_sensing_fsm_create(&cfg, &s_fsm));
    ESP_ERROR_CHECK(esp_wifi_sensing_fsm_add_channel(s_fsm, s_ap_mac));
    if (s_has_pair) ESP_ERROR_CHECK(esp_wifi_sensing_fsm_add_channel(s_fsm, s_pair_mac));
    ESP_ERROR_CHECK(esp_wifi_sensing_fsm_register_event_cb(s_fsm, ESP_WIFI_SENSING_FSM_EVENT_ACTIVE, on_event, NULL));
    ESP_ERROR_CHECK(esp_wifi_sensing_fsm_register_event_cb(s_fsm, ESP_WIFI_SENSING_FSM_EVENT_INACTIVE, on_event, NULL));
    ESP_ERROR_CHECK(esp_wifi_sensing_fsm_control(s_fsm, ESP_WIFI_SENSING_FSM_CTRL_START, NULL));

    esp_err_t pe = esp_wifi_sensing_fsm_ping_router_start(s_fsm);
    if (pe != ESP_OK) ESP_LOGW(TAG, "router ping not started: %s", esp_err_to_name(pe));

    s_snap.running = true;
    s_train_running = false;
    s_train_complete = false;
    xTaskCreate(logic_task, "sense_logic", 4096, NULL, 4, NULL);
    ESP_LOGI(TAG, "sensing started (pair=%s, ping=%d Hz)", s_has_pair ? "yes" : "no", c->ping_hz);
    return ESP_OK;
}

static void sender_task(void *arg)
{
    uint32_t counter = 0;
    for (;;) {
        uint8_t hz = settings_get()->ping_hz;
        if (hz < 5) hz = 5;
        esp_now_send(BROADCAST, (uint8_t *)&counter, sizeof(counter));
        counter++;
        vTaskDelay(pdMS_TO_TICKS(1000 / hz));
    }
}

esp_err_t sender_start(void)
{
    esp_err_t e = esp_now_init();
    if (e != ESP_OK && e != ESP_ERR_ESPNOW_EXIST) return e;
    esp_now_peer_info_t p = {0};
    memcpy(p.peer_addr, BROADCAST, 6);
    p.channel = 0;
    p.ifidx = WIFI_IF_STA;
    p.encrypt = false;
    if (!esp_now_is_peer_exist(BROADCAST)) esp_now_add_peer(&p);
    xTaskCreate(sender_task, "sender", 3072, NULL, 4, NULL);
    ESP_LOGI(TAG, "sender mode: transmitting for the receiver board");
    return ESP_OK;
}

void sensing_apply_settings(void)
{
    if (!s_fsm) return;
    settings_t *c = settings_get();
    const uint8_t *macs[2] = {s_ap_mac, s_pair_mac};
    for (int i = 0; i < (s_has_pair ? 2 : 1); i++) {
        esp_wifi_sensing_fsm_channel_config_t cc = {0};
        if (esp_wifi_sensing_fsm_get_channel_config(s_fsm, macs[i], &cc) != ESP_OK) continue;
        cc.sensitivity = c->sensitivity;
        cc.presence_sensitivity = c->presence_sensitivity;
        cc.active_jitter_min = c->jitter_min;
        cc.active_filter_ms = c->filter_ms;
        esp_wifi_sensing_fsm_set_channel_config(s_fsm, macs[i], &cc);
    }
}

void sensing_get(sensing_snapshot_t *out) { *out = s_snap; }

esp_err_t sensing_train_start(void)
{
    if (!s_fsm) return ESP_ERR_INVALID_STATE;
    if (s_train_running) return ESP_ERR_INVALID_STATE;
    esp_err_t e = esp_wifi_sensing_fsm_train_start(s_fsm, primary_mac());
    if (e != ESP_OK) return e;
    s_train_started_us = esp_timer_get_time();
    s_train_running = true;
    s_train_complete = false;
    s_snap.train_elapsed_s = 0;
    s_snap.train_remaining_s = ROOM_CALIBRATION_SECONDS;
    (void)door_sensor_reset_baseline();
    ESP_LOGI(TAG, "room calibration started for 600 seconds");
    return ESP_OK;
}

esp_err_t sensing_train_stop(float *w, float *j)
{
    if (!s_fsm || !s_train_running) return ESP_ERR_INVALID_STATE;
    s_train_running = false;
    esp_err_t e = esp_wifi_sensing_fsm_train_stop(s_fsm, primary_mac(), w, j);
    if (e == ESP_OK) {
        s_snap.train_elapsed_s = (uint16_t)((esp_timer_get_time() - s_train_started_us) / 1000000LL);
        s_train_complete = s_snap.train_elapsed_s >= ROOM_CALIBRATION_SECONDS;
        s_snap.trained_wander = w ? *w : 0.0f;
        s_snap.trained_jitter = j ? *j : 0.0f;
    }
    s_snap.train_running = false;
    s_snap.train_complete = s_train_complete;
    s_snap.train_remaining_s = 0;
    return e;
}

esp_err_t sensing_train_remove(void)
{
    if (!s_fsm) return ESP_ERR_INVALID_STATE;
    s_train_running = false;
    s_train_complete = false;
    s_snap.train_running = false;
    s_snap.train_complete = false;
    s_snap.train_elapsed_s = 0;
    s_snap.train_remaining_s = 0;
    return esp_wifi_sensing_fsm_train_remove(s_fsm, primary_mac());
}

esp_err_t sensing_reset_baseline(void)
{
    return s_fsm ? esp_wifi_sensing_fsm_control(s_fsm, ESP_WIFI_SENSING_FSM_CTRL_RESET_BASELINE, NULL) : ESP_ERR_INVALID_STATE;
}
