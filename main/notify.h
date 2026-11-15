#pragma once
#include <stdbool.h>

typedef enum { NOTIFY_MOTION = 0, NOTIFY_PRESENCE = 1, NOTIFY_DOOR_CROSSING = 2, NOTIFY_DOOR_SENSOR_FAULT = 3 } notify_type_t;

void notify_start(void);
void notify_reconfigure(void);
void notify_post(notify_type_t type, bool value);
bool notify_mqtt_connected(void);
