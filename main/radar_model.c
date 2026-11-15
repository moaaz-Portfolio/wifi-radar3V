#include "radar_model.h"
#include <math.h>
#include <string.h>

#define T RM_WIN
#define CH 128

static const uint32_t EXPECT[39][2] = {
    {32, 384}, {32, 1}, {32, 640}, {32, 1}, {32, 1152}, {32, 1},
    {96, 480}, {96, 1}, {96, 480}, {96, 1}, {24, 96}, {24, 1}, {96, 24}, {96, 1},
    {128, 480}, {128, 1}, {128, 640}, {128, 1}, {32, 128}, {32, 1}, {128, 32}, {128, 1},
    {128, 96},
    {256, 128}, {256, 64}, {256, 1}, {256, 128}, {256, 64}, {256, 1},
    {32, 128}, {32, 1}, {1, 32}, {1, 1}, {64, 384}, {64, 1}, {64, 1}, {64, 1}, {4, 64}, {4, 1}};

enum {
    S0W, S0B, S1W, S1B, S2W, S2B,
    B1C1W, B1C1B, B1C2W, B1C2B, B1F1W, B1F1B, B1F2W, B1F2B,
    B2C1W, B2C1B, B2C2W, B2C2B, B2F1W, B2F1B, B2F2W, B2F2B,
    SKIPW,
    LFWIH, LFWHH, LFB, LRWIH, LRWHH, LRB,
    A1W, A1B, A2W, A2B, EMW, EMB, LNW, LNB, CLW, CLB
};

static uint32_t rd32(const uint8_t *p)
{
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static float rdf(const uint8_t *p)
{
    float v;
    memcpy(&v, p, 4);
    return v;
}

size_t rm_work_floats(void)
{
    return 3u * T * CH;
}

int rm_init(rm_model_t *m, const uint8_t *blob, size_t len, float *work)
{
    memset(m, 0, sizeof(*m));
    if (len < 24 || memcmp(blob, "RMD1", 4) != 0 || rd32(blob + 4) != 1 || rd32(blob + 8) != 39) return -1;
    m->in_scale = rdf(blob + 12);
    m->temperature = rdf(blob + 16);
    m->threshold = rdf(blob + 20);
    size_t off = 24;
    for (int i = 0; i < 39; i++) {
        if (off + 12 > len) return -2;
        rm_tensor_t *t = &m->t[i];
        t->kind = rd32(blob + off);
        t->rows = rd32(blob + off + 4);
        t->cols = rd32(blob + off + 8);
        off += 12;
        if (t->rows != EXPECT[i][0] || t->cols != EXPECT[i][1]) return -3;
        size_t n = (size_t)t->rows * t->cols;
        if (t->kind == 1) {
            if (off + 4u * t->rows + n > len) return -2;
            t->scale = (const float *)(blob + off);
            off += 4u * t->rows;
            t->q = (const int8_t *)(blob + off);
            off += n;
            off = (off + 3u) & ~3u;
        } else if (t->kind == 0) {
            if (off + 4u * n > len) return -2;
            t->f = (const float *)(blob + off);
            off += 4u * n;
        } else {
            return -4;
        }
    }
    m->buf_p = work;
    m->buf_q = work + T * CH;
    m->buf_r = work + 2 * T * CH;
    m->ready = 1;
    return 0;
}

static float dot_q(const int8_t *w, const float *x, int n)
{
    float a = 0.0f, b = 0.0f;
    int i = 0;
    for (; i + 1 < n; i += 2) {
        a += (float)w[i] * x[i];
        b += (float)w[i + 1] * x[i + 1];
    }
    if (i < n) a += (float)w[i] * x[i];
    return a + b;
}

static float dot_f(const float *w, const float *x, int n)
{
    float a = 0.0f;
    for (int i = 0; i < n; i++) a += w[i] * x[i];
    return a;
}

static float row_dot(const rm_tensor_t *t, int r, int col0, const float *x, int n)
{
    if (t->kind == 1) return dot_q(t->q + (size_t)r * t->cols + col0, x, n);
    return dot_f(t->f + (size_t)r * t->cols + col0, x, n);
}

static float row_scale(const rm_tensor_t *t, int r)
{
    return t->kind == 1 ? t->scale[r] : 1.0f;
}

static float bias_at(const rm_tensor_t *b, int r)
{
    return b->f[r];
}

static float gelu(float x)
{
    return 0.5f * x * (1.0f + erff(x * 0.70710678118f));
}

static float sigm(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static void conv_tm(const float *in, int cin, const rm_tensor_t *w, const rm_tensor_t *b,
                    float *out, int out_stride, int out_off, int K, int dil, int pad)
{
    int cout = (int)w->rows;
    for (int t = 0; t < T; t++) {
        for (int o = 0; o < cout; o++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                int tp = t + k * dil - pad;
                if (tp < 0 || tp >= T) continue;
                acc += row_dot(w, o, k * cin, in + (size_t)tp * cin, cin);
            }
            float v = acc * row_scale(w, o);
            if (b) v += bias_at(b, o);
            out[(size_t)t * out_stride + out_off + o] = v;
        }
    }
}

static void gelu_inplace(float *x, int n)
{
    for (int i = 0; i < n; i++) x[i] = gelu(x[i]);
}

static void se_apply(float *h, int c, const rm_tensor_t *f1w, const rm_tensor_t *f1b,
                     const rm_tensor_t *f2w, const rm_tensor_t *f2b)
{
    float g[CH], u[64], w[CH];
    for (int j = 0; j < c; j++) g[j] = 0.0f;
    for (int t = 0; t < T; t++)
        for (int j = 0; j < c; j++) g[j] += h[(size_t)t * c + j];
    for (int j = 0; j < c; j++) g[j] /= (float)T;
    int r = (int)f1w->rows;
    for (int i = 0; i < r; i++) u[i] = gelu(row_dot(f1w, i, 0, g, c) * row_scale(f1w, i) + bias_at(f1b, i));
    for (int j = 0; j < c; j++) w[j] = sigm(row_dot(f2w, j, 0, u, r) * row_scale(f2w, j) + bias_at(f2b, j));
    for (int t = 0; t < T; t++)
        for (int j = 0; j < c; j++) h[(size_t)t * c + j] *= w[j];
}

static void lstm_dir(const rm_model_t *m, const float *x, int wih, int whh, int bi, int reverse,
                     float *out, int out_off)
{
    float h[64] = {0}, c[64] = {0}, g[256];
    const rm_tensor_t *a = &m->t[wih], *b = &m->t[whh], *bb = &m->t[bi];
    for (int s = 0; s < T; s++) {
        int t = reverse ? T - 1 - s : s;
        const float *xt = x + (size_t)t * CH;
        for (int r = 0; r < 256; r++)
            g[r] = bias_at(bb, r) + row_dot(a, r, 0, xt, CH) * row_scale(a, r) + row_dot(b, r, 0, h, 64) * row_scale(b, r);
        for (int j = 0; j < 64; j++) {
            float ig = sigm(g[j]), fg = sigm(g[64 + j]), gg = tanhf(g[128 + j]), og = sigm(g[192 + j]);
            c[j] = fg * c[j] + ig * gg;
            h[j] = og * tanhf(c[j]);
        }
        for (int j = 0; j < 64; j++) out[(size_t)t * CH + out_off + j] = h[j];
    }
}

void rm_forward(rm_model_t *m, const float *win, float *logits)
{
    float *P = m->buf_p, *Q = m->buf_q, *R = m->buf_r;
    const rm_tensor_t *t = m->t;

    for (int s = 0; s < T; s++) {
        for (int f = 0; f < RM_FEATURES; f++) {
            float v = win[s * RM_FEATURES + f];
            P[s * 128 + f] = v;
            P[s * 128 + RM_FEATURES + f] = s == 0 ? 0.0f : v - win[(s - 1) * RM_FEATURES + f];
        }
    }
    conv_tm(P, 128, &t[S0W], &t[S0B], Q, 96, 0, 3, 1, 1);
    conv_tm(P, 128, &t[S1W], &t[S1B], Q, 96, 32, 5, 1, 2);
    conv_tm(P, 128, &t[S2W], &t[S2B], Q, 96, 64, 9, 1, 4);
    gelu_inplace(Q, T * 96);

    conv_tm(Q, 96, &t[B1C1W], &t[B1C1B], R, 96, 0, 5, 1, 2);
    gelu_inplace(R, T * 96);
    conv_tm(R, 96, &t[B1C2W], &t[B1C2B], P, 96, 0, 5, 1, 2);
    se_apply(P, 96, &t[B1F1W], &t[B1F1B], &t[B1F2W], &t[B1F2B]);
    for (int i = 0; i < T * 96; i++) Q[i] = gelu(P[i] + Q[i]);

    conv_tm(Q, 96, &t[B2C1W], &t[B2C1B], R, 128, 0, 5, 2, 4);
    gelu_inplace(R, T * 128);
    conv_tm(R, 128, &t[B2C2W], &t[B2C2B], P, 128, 0, 5, 2, 4);
    se_apply(P, 128, &t[B2F1W], &t[B2F1B], &t[B2F2W], &t[B2F2B]);
    conv_tm(Q, 96, &t[SKIPW], NULL, R, 128, 0, 1, 1, 0);
    for (int i = 0; i < T * 128; i++) P[i] = gelu(P[i] + R[i]);

    lstm_dir(m, P, LFWIH, LFWHH, LFB, 0, Q, 0);
    lstm_dir(m, P, LRWIH, LRWHH, LRB, 1, Q, 64);

    float att[T], mx = -1e30f, sum = 0.0f;
    for (int s = 0; s < T; s++) {
        float u[32];
        const float *h = Q + (size_t)s * CH;
        for (int i = 0; i < 32; i++) u[i] = tanhf(row_dot(&t[A1W], i, 0, h, CH) * row_scale(&t[A1W], i) + bias_at(&t[A1B], i));
        att[s] = row_dot(&t[A2W], 0, 0, u, 32) * row_scale(&t[A2W], 0) + bias_at(&t[A2B], 0);
        if (att[s] > mx) mx = att[s];
    }
    for (int s = 0; s < T; s++) {
        att[s] = expf(att[s] - mx);
        sum += att[s];
    }
    float pooled[384];
    for (int j = 0; j < CH; j++) {
        float wsum = 0.0f, mean = 0.0f, mxv = -1e30f;
        for (int s = 0; s < T; s++) {
            float v = Q[(size_t)s * CH + j];
            wsum += att[s] / sum * v;
            mean += v;
            if (v > mxv) mxv = v;
        }
        pooled[j] = wsum;
        pooled[CH + j] = mean / (float)T;
        pooled[2 * CH + j] = mxv;
    }
    float e[64], mu = 0.0f, var = 0.0f;
    for (int i = 0; i < 64; i++) {
        e[i] = row_dot(&t[EMW], i, 0, pooled, 384) * row_scale(&t[EMW], i) + bias_at(&t[EMB], i);
        mu += e[i];
    }
    mu /= 64.0f;
    for (int i = 0; i < 64; i++) var += (e[i] - mu) * (e[i] - mu);
    var /= 64.0f;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < 64; i++) e[i] = gelu((e[i] - mu) * inv * bias_at(&t[LNW], i) + bias_at(&t[LNB], i));
    for (int k = 0; k < RM_CLASSES; k++)
        logits[k] = row_dot(&t[CLW], k, 0, e, 64) * row_scale(&t[CLW], k) + bias_at(&t[CLB], k);
}

int rm_build_window(const uint32_t *ts, const uint8_t *amp, int n, float in_scale, float *out)
{
    if (n < 2) return 0;
    double tl = (double)ts[n - 1];
    double g0 = tl - 1000.0 / RM_FS * (RM_WIN - 1);
    if (g0 < (double)ts[0]) return 0;
    int i = 0;
    for (int j = 0; j < RM_WIN; j++) {
        double g = tl - 1000.0 / RM_FS * (RM_WIN - 1 - j);
        while (i < n - 2 && (double)ts[i + 1] < g) i++;
        double t0 = ts[i], t1 = ts[i + 1];
        double a = t1 > t0 ? (g - t0) / (t1 - t0) : 1.0;
        if (a < 0.0) a = 0.0;
        if (a > 1.0) a = 1.0;
        for (int f = 0; f < RM_FEATURES; f++) {
            double v0 = amp[(size_t)i * RM_FEATURES + f], v1 = amp[(size_t)(i + 1) * RM_FEATURES + f];
            out[j * RM_FEATURES + f] = (float)(v0 + (v1 - v0) * a);
        }
    }
    for (int f = 0; f < RM_FEATURES; f++) {
        float mean = 0.0f;
        for (int j = 0; j < RM_WIN; j++) mean += out[j * RM_FEATURES + f];
        mean /= (float)RM_WIN;
        for (int j = 0; j < RM_WIN; j++) out[j * RM_FEATURES + f] = (out[j * RM_FEATURES + f] - mean) / in_scale;
    }
    return 1;
}

void rm_decider_reset(rm_decider_t *d)
{
    memset(d, 0, sizeof(*d));
    d->current = -1;
    d->pending = -1;
}

static void softmax_n(const float *z, float *p, int n)
{
    float mx = z[0], s = 0.0f;
    for (int i = 1; i < n; i++) if (z[i] > mx) mx = z[i];
    for (int i = 0; i < n; i++) {
        p[i] = expf(z[i] - mx);
        s += p[i];
    }
    for (int i = 0; i < n; i++) p[i] /= s;
}

void rm_decide(const rm_model_t *m, rm_decider_t *d, const float *logits, rm_result_t *r)
{
    float z[RM_CLASSES], p[RM_CLASSES];
    for (int i = 0; i < RM_CLASSES; i++) z[i] = logits[i] / m->temperature;
    softmax_n(z, p, RM_CLASSES);
    int slot = d->count % 4;
    for (int i = 0; i < RM_CLASSES; i++) d->hist[slot][i] = logf(p[i] + 1e-12f);
    d->count++;
    int n = d->count < 4 ? d->count : 4;
    float mean[RM_CLASSES] = {0}, avg[RM_CLASSES];
    for (int k = 0; k < n; k++)
        for (int i = 0; i < RM_CLASSES; i++) mean[i] += d->hist[k][i] / (float)n;
    softmax_n(mean, avg, RM_CLASSES);
    int top = 0;
    for (int i = 1; i < RM_CLASSES; i++) if (avg[i] > avg[top]) top = i;
    if (d->current < 0) {
        d->current = top;
    } else if (top != d->current) {
        if (d->pending == top) d->pending_count++;
        else { d->pending = top; d->pending_count = 1; }
        if (d->pending_count >= 2) { d->current = top; d->pending = -1; d->pending_count = 0; }
    } else {
        d->pending = -1;
        d->pending_count = 0;
    }
    r->best = d->current;
    r->confidence = avg[d->current];
    r->label = r->confidence >= m->threshold ? d->current : -1;
    for (int i = 0; i < RM_CLASSES; i++) r->probs[i] = avg[i];
}
