#include "radar_ai.h"
#include <math.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "ping/ping_sock.h"

#include "radar_model.h"
#include "settings.h"
#include "outputs.h"

static const char *TAG = "radar_ai";

#define CSI_SUBCARRIERS 64
#define RING_N 256
#define SPAN_MS 1500
#define MIN_GAP_MS 5

static uint32_t s_ts[RING_N];
static uint8_t s_amp[RING_N][CSI_SUBCARRIERS];
static int s_head = 0;
static int s_count = 0;
static portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;
static uint8_t s_bssid[6];

static uint32_t c_ts[RING_N];
static uint8_t c_amp[RING_N][CSI_SUBCARRIERS];
static float s_win[RM_WIN * RM_FEATURES];

static rm_model_t s_model;
static rm_decider_t s_decider;
static radar_ai_snapshot_t s_snap;

static void set_error(const char *msg)
{
    portENTER_CRITICAL(&s_mux);
    s_snap.error = msg;
    portEXIT_CRITICAL(&s_mux);
}

static void csi_cb(void *ctx, wifi_csi_info_t *info)
{
    (void)ctx;
    if (!info || !info->buf || info->len < 2 * CSI_SUBCARRIERS) return;
    if (memcmp(info->mac, s_bssid, 6) != 0) return;

    uint32_t now = (uint32_t)(esp_timer_get_time() / 1000);
    uint8_t row[CSI_SUBCARRIERS];
    const int8_t *b = (const int8_t *)info->buf;
    for (int i = 0; i < CSI_SUBCARRIERS; i++) {
        float im = b[2 * i], re = b[2 * i + 1];
        float a = sqrtf(re * re + im * im) + 0.5f;
        row[i] = a > 255.0f ? 255 : (uint8_t)a;
    }

    portENTER_CRITICAL(&s_mux);
    int last = (s_head + RING_N - 1) % RING_N;
    if (s_count == 0 || now - s_ts[last] >= MIN_GAP_MS) {
        s_ts[s_head] = now;
        memcpy(s_amp[s_head], row, CSI_SUBCARRIERS);
        s_head = (s_head + 1) % RING_N;
        if (s_count < RING_N) s_count++;
    }
    portEXIT_CRITICAL(&s_mux);
}

static int snapshot_frames(void)
{
    int n = 0;
    portENTER_CRITICAL(&s_mux);
    if (s_count > 1) {
        int start = (s_head + RING_N - s_count) % RING_N;
        int last = (s_head + RING_N - 1) % RING_N;
        uint32_t t_last = s_ts[last];
        int first = 0;
        for (int k = s_count - 1; k >= 0; k--) {
            int idx = (start + k) % RING_N;
            if (t_last - s_ts[idx] > SPAN_MS) { first = k + 1; break; }
        }
        for (int k = first; k < s_count; k++) {
            int idx = (start + k) % RING_N;
            c_ts[n] = s_ts[idx];
            memcpy(c_amp[n], s_amp[idx], CSI_SUBCARRIERS);
            n++;
        }
    }
    portEXIT_CRITICAL(&s_mux);
    return n;
}

static void ai_task(void *arg)
{
    (void)arg;
    float logits[RM_CLASSES];
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(200));
        int n = snapshot_frames();
        if (n < 2) continue;
        if (!rm_build_window(c_ts, &c_amp[0][0], n, s_model.in_scale, s_win)) continue;

        int64_t t0 = esp_timer_get_time();
        rm_forward(&s_model, s_win, logits);
        uint32_t ms = (uint32_t)((esp_timer_get_time() - t0) / 1000);

        rm_result_t r;
        rm_decide(&s_model, &s_decider, logits, &r);

        portENTER_CRITICAL(&s_mux);
        s_snap.ready = true;
        s_snap.label = r.label;
        s_snap.best = r.best;
        s_snap.confidence = r.confidence;
        memcpy(s_snap.probs, r.probs, sizeof(r.probs));
        s_snap.infer_ms = ms;
        s_snap.count++;
        portEXIT_CRITICAL(&s_mux);
    }
}

static void start_ping(uint8_t hz)
{
    esp_netif_t *sta = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
    esp_netif_ip_info_t ip;
    if (!sta || esp_netif_get_ip_info(sta, &ip) != ESP_OK || ip.gw.addr == 0) {
        ESP_LOGE(TAG, "no gateway to ping");
        return;
    }
    esp_ping_config_t pc = ESP_PING_DEFAULT_CONFIG();
    pc.target_addr.type = IPADDR_TYPE_V4;
    pc.target_addr.u_addr.ip4.addr = ip.gw.addr;
    pc.count = ESP_PING_COUNT_INFINITE;
    pc.interval_ms = 1000 / (hz ? hz : 50);
    pc.timeout_ms = 500;

    esp_ping_callbacks_t cbs = {0};
    esp_ping_handle_t h = NULL;
    if (esp_ping_new_session(&pc, &cbs, &h) == ESP_OK) esp_ping_start(h);
    else ESP_LOGE(TAG, "ping session failed");
}

void radar_ai_get(radar_ai_snapshot_t *out)
{
    portENTER_CRITICAL(&s_mux);
    *out = s_snap;
    portEXIT_CRITICAL(&s_mux);
}

esp_err_t radar_ai_start(void)
{
    memset(&s_snap, 0, sizeof(s_snap));
    s_snap.label = -1;
    s_snap.best = -1;
    rm_decider_reset(&s_decider);

    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    if (!part) {
        set_error("model partition missing");
        ESP_LOGE(TAG, "model partition missing");
        return ESP_ERR_NOT_FOUND;
    }

    const void *blob = NULL;
    esp_partition_mmap_handle_t handle;
    if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &blob, &handle) != ESP_OK) {
        set_error("model mmap failed");
        return ESP_FAIL;
    }

    size_t bytes = rm_work_floats() * sizeof(float);
    float *work = heap_caps_malloc(bytes, MALLOC_CAP_8BIT);
    if (!work) work = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM);
    if (!work) {
        set_error("not enough memory");
        return ESP_ERR_NO_MEM;
    }

    int rc = rm_init(&s_model, (const uint8_t *)blob, part->size, work);
    if (rc != 0) {
        set_error("model file invalid");
        ESP_LOGE(TAG, "model invalid (%d)", rc);
        return ESP_ERR_INVALID_STATE;
    }
    s_snap.model_ok = true;

    wifi_ap_record_t ap;
    esp_err_t e = esp_wifi_sta_get_ap_info(&ap);
    if (e != ESP_OK) return e;
    memcpy(s_bssid, ap.bssid, 6);

    wifi_csi_config_t cfg = {
        .lltf_en = true,
        .htltf_en = false,
        .stbc_htltf2_en = false,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = 0,
        .dump_ack_en = false,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    xTaskCreate(ai_task, "radar_ai", 8192, NULL, 3, NULL);
    start_ping(settings_get()->ping_hz);
    outputs_set_motion(true);
    ESP_LOGI(TAG, "on-board AI started");
    return ESP_OK;
}
