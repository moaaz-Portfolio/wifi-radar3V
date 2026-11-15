#include "door_sensor.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "esp_timer.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"

#include "settings.h"
#include "notify.h"

#define DOOR_TRIG_GPIO GPIO_NUM_4
#define DOOR_ECHO_GPIO GPIO_NUM_5
#define BUZZER_GPIO GPIO_NUM_6
#define ECHO_TIMEOUT_US 30000
#define SAMPLE_DELAY_MS 80
#define UDP_PACKET_INTERVAL_MS 500

static const char *TAG = "door_sensor";
static portMUX_TYPE s_lock = portMUX_INITIALIZER_UNLOCKED;
static door_sensor_snapshot_t s_state;
static bool s_started;
static volatile bool s_relearn_baseline;

typedef struct __attribute__((packed)) {
    char magic[4];
    uint32_t sequence;
    uint16_t distance_cm;
    uint8_t present;
    uint8_t sensor_ok;
    uint8_t buzzer_active;
    uint8_t crossing;
    uint8_t baseline_valid;
    uint8_t sensor_fault;
    uint16_t baseline_cm;
} door_packet_t;

void door_sensor_get(door_sensor_snapshot_t *out)
{
    if (!out) return;
    portENTER_CRITICAL(&s_lock);
    *out = s_state;
    portEXIT_CRITICAL(&s_lock);
}

esp_err_t door_sensor_reset_baseline(void)
{
    settings_t *cfg = settings_get();
    cfg->door_baseline_cm = 0;
    cfg->door_baseline_valid = false;
    esp_err_t err = settings_save();
    if (err != ESP_OK) return err;
    s_relearn_baseline = true;
    return ESP_OK;
}

static uint16_t median_u16(uint16_t *v, size_t n)
{
    uint16_t a[30];
    if (n > 30) n = 30;
    memcpy(a, v, n * sizeof(uint16_t));
    for (size_t i = 1; i < n; i++) {
        uint16_t x = a[i]; size_t j = i;
        while (j && a[j - 1] > x) { a[j] = a[j - 1]; j--; }
        a[j] = x;
    }
    return a[n / 2];
}

static bool read_distance(uint16_t *distance_cm)
{
    gpio_set_level(DOOR_TRIG_GPIO, 0);
    esp_rom_delay_us(3);
    gpio_set_level(DOOR_TRIG_GPIO, 1);
    esp_rom_delay_us(10);
    gpio_set_level(DOOR_TRIG_GPIO, 0);

    int64_t rise_deadline = esp_timer_get_time() + ECHO_TIMEOUT_US;
    while (gpio_get_level(DOOR_ECHO_GPIO) == 0) {
        if (esp_timer_get_time() >= rise_deadline) return false;
    }

    int64_t pulse_start = esp_timer_get_time();
    int64_t pulse_deadline = pulse_start + ECHO_TIMEOUT_US;
    while (gpio_get_level(DOOR_ECHO_GPIO) == 1) {
        if (esp_timer_get_time() >= pulse_deadline) return false;
    }

    int64_t pulse_us = esp_timer_get_time() - pulse_start;
    if (pulse_us <= 0 || pulse_us > ECHO_TIMEOUT_US) return false;

    float cm = ((float)pulse_us * 0.0343f) / 2.0f;
    if (cm < 2.0f || cm > 400.0f) return false;
    *distance_cm = (uint16_t)(cm + 0.5f);
    return true;
}

static void publish_udp(int sock, uint32_t sequence, const door_sensor_snapshot_t *state)
{
    if (sock < 0) return;

    const settings_t *cfg = settings_get();
    uint16_t stream_port = cfg->stream_port ? cfg->stream_port : 5005;
    uint16_t telemetry_port = stream_port < 65535 ? (uint16_t)(stream_port + 1) : 5006;

    struct sockaddr_in dest = {0};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(telemetry_port);
    if (cfg->stream_host[0] == 0 ||
        inet_pton(AF_INET, cfg->stream_host, &dest.sin_addr) != 1) {
        dest.sin_addr.s_addr = htonl(INADDR_BROADCAST);
    }

    door_packet_t packet = {
        .magic = {'D', 'O', 'R', '2'},
        .sequence = sequence,
        .distance_cm = state->distance_valid ? state->distance_cm : 0,
        .present = state->present ? 1 : 0,
        .sensor_ok = state->sensor_ok ? 1 : 0,
        .buzzer_active = state->buzzer_active ? 1 : 0,
        .crossing = state->crossing ? 1 : 0,
        .baseline_valid = state->baseline_valid ? 1 : 0,
        .sensor_fault = state->sensor_fault ? 1 : 0,
        .baseline_cm = state->baseline_cm,
    };
    (void)sendto(sock, &packet, sizeof(packet), 0,
                 (struct sockaddr *)&dest, sizeof(dest));
}

static void door_task(void *arg)
{
    (void)arg;
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock >= 0) {
        int enabled = 1;
        (void)setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &enabled, sizeof(enabled));
    }

    uint32_t sequence = 0;
    uint32_t tick = 0;
    uint32_t last_publish_tick = UINT32_MAX;
    unsigned near_count = 0;
    unsigned far_count = 0;
    bool present = false, crossing = false, previous_fault = false;
    bool had_valid_read = false;
    uint16_t baseline_samples[30] = {0};
    size_t baseline_count = 0;
    unsigned invalid_count = 0;
    int64_t crossing_until_us = 0;

    for (;;) {
        const settings_t *cfg = settings_get();
        door_sensor_snapshot_t next = {0};
        next.enabled = cfg->door_enabled;
        next.baseline_valid = cfg->door_baseline_valid;
        next.baseline_cm = cfg->door_baseline_cm;

        if (s_relearn_baseline) {
            s_relearn_baseline = false;
            baseline_count = 0;
            invalid_count = 0;
            near_count = far_count = 0;
            present = crossing = false;
            crossing_until_us = 0;
        }

        uint16_t distance = 0;
        bool valid = cfg->door_enabled && read_distance(&distance);
        next.distance_valid = valid;
        next.distance_cm = valid ? distance : 0;

        if (!cfg->door_enabled) {
            near_count = far_count = invalid_count = baseline_count = 0;
            present = crossing = false;
            next.sensor_ok = false;
            next.sensor_fault = false;
        } else {
            if (valid) { invalid_count = 0; had_valid_read = true; }
            else if (invalid_count < 10) invalid_count++;
            next.sensor_fault = invalid_count >= 10;
            next.sensor_ok = !next.sensor_fault;

            if (valid && !cfg->door_baseline_valid && baseline_count < 30) {
                baseline_samples[baseline_count++] = distance;
                if (baseline_count == 30) {
                    settings_t *mutable_cfg = settings_get();
                    mutable_cfg->door_baseline_cm = median_u16(baseline_samples, baseline_count);
                    mutable_cfg->door_baseline_valid = true;
                    if (settings_save() != ESP_OK) {
                        mutable_cfg->door_baseline_valid = false;
                        ESP_LOGW(TAG, "could not save clear-path baseline");
                    } else {
                        ESP_LOGI(TAG, "clear-path baseline learned: %u cm", mutable_cfg->door_baseline_cm);
                    }
                }
            }

            const settings_t *current = settings_get();
            next.baseline_valid = current->door_baseline_valid;
            next.baseline_cm = current->door_baseline_cm;
            if (!valid && had_valid_read && next.baseline_valid &&
                invalid_count >= 3 && invalid_count < 10 && !crossing) {
                present = true;
                crossing = true;
                crossing_until_us = esp_timer_get_time() + 5000000LL;
                notify_post(NOTIFY_DOOR_CROSSING, true);
            }

            if (valid && next.baseline_valid && !next.sensor_fault) {
                uint16_t margin = next.baseline_cm / 7;
                if (margin < 18) margin = 18;
                uint16_t trigger_cm = next.baseline_cm > margin ? next.baseline_cm - margin : 0;
                if (!present && distance <= trigger_cm) {
                    near_count++;
                    far_count = 0;
                    if (near_count >= 2) {
                        present = true;
                        crossing = true;
                        crossing_until_us = esp_timer_get_time() + 5000000LL;
                        notify_post(NOTIFY_DOOR_CROSSING, true);
                    }
                } else if (present && distance >= trigger_cm + margin / 2) {
                    far_count++;
                    near_count = 0;
                    if (far_count >= 3) present = false;
                } else {
                    near_count = far_count = 0;
                }
            } else if (next.sensor_fault) {
                present = false;
                near_count = far_count = 0;
            }
            if (crossing && esp_timer_get_time() >= crossing_until_us) {
                crossing = false;
                notify_post(NOTIFY_DOOR_CROSSING, false);
            }
        }

        if (next.sensor_fault != previous_fault) {
            notify_post(NOTIFY_DOOR_SENSOR_FAULT, next.sensor_fault);
            previous_fault = next.sensor_fault;
        }
        next.present = present;
        next.crossing = crossing;
        next.buzzer_active = cfg->door_enabled && cfg->buzzer_enabled &&
                             crossing && (tick % 9 < 2);
        gpio_set_level(BUZZER_GPIO, next.buzzer_active ? 1 : 0);

        portENTER_CRITICAL(&s_lock);
        s_state = next;
        portEXIT_CRITICAL(&s_lock);

        if (last_publish_tick == UINT32_MAX ||
            (tick - last_publish_tick) * SAMPLE_DELAY_MS >= UDP_PACKET_INTERVAL_MS) {
            publish_udp(sock, sequence++, &next);
            last_publish_tick = tick;
        }

        tick++;
        vTaskDelay(pdMS_TO_TICKS(SAMPLE_DELAY_MS));
    }
}

esp_err_t door_sensor_start(void)
{
    if (s_started) return ESP_ERR_INVALID_STATE;

    gpio_config_t trig = {
        .pin_bit_mask = 1ULL << DOOR_TRIG_GPIO,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    esp_err_t err = gpio_config(&trig);
    if (err != ESP_OK) return err;
    gpio_set_level(DOOR_TRIG_GPIO, 0);

    gpio_config_t echo = {
        .pin_bit_mask = 1ULL << DOOR_ECHO_GPIO,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    err = gpio_config(&echo);
    if (err != ESP_OK) return err;

    gpio_config_t buzzer = {
        .pin_bit_mask = 1ULL << BUZZER_GPIO,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    err = gpio_config(&buzzer);
    if (err != ESP_OK) return err;
    gpio_set_level(BUZZER_GPIO, 0);

    s_started = true;
    BaseType_t task_ok = xTaskCreate(door_task, "door_sensor", 4096, NULL, 3, NULL);
    if (task_ok != pdPASS) {
        s_started = false;
        return ESP_ERR_NO_MEM;
    }

    ESP_LOGI(TAG, "HC-SR04 TRIG=%d ECHO=%d, buzzer driver=%d",
             DOOR_TRIG_GPIO, DOOR_ECHO_GPIO, BUZZER_GPIO);
    return ESP_OK;
}
