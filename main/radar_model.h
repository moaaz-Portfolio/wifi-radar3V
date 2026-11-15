#pragma once
#include <stdint.h>
#include <stddef.h>

#define RM_FEATURES 64
#define RM_WIN 64
#define RM_CLASSES 4
#define RM_FS 50

typedef struct {
    uint32_t kind;
    uint32_t rows;
    uint32_t cols;
    const float *scale;
    const int8_t *q;
    const float *f;
} rm_tensor_t;

typedef struct {
    int ready;
    float in_scale;
    float temperature;
    float threshold;
    rm_tensor_t t[39];
    float *buf_p;
    float *buf_q;
    float *buf_r;
} rm_model_t;

typedef struct {
    float hist[4][RM_CLASSES];
    int count;
    int current;
    int pending;
    int pending_count;
} rm_decider_t;

typedef struct {
    int best;
    int label;
    float confidence;
    float probs[RM_CLASSES];
} rm_result_t;

int rm_init(rm_model_t *m, const uint8_t *blob, size_t len, float *work);
size_t rm_work_floats(void);
void rm_forward(rm_model_t *m, const float *win, float *logits);

int rm_build_window(const uint32_t *ts, const uint8_t *amp, int n, float in_scale, float *out);

void rm_decider_reset(rm_decider_t *d);
void rm_decide(const rm_model_t *m, rm_decider_t *d, const float *logits, rm_result_t *r);
