import numpy as np

EPS = 1e-12


def softmax(z, axis=-1):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def nll(logits, y, temperature=1.0):
    p = softmax(np.asarray(logits, dtype=np.float64) / temperature)
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + EPS)))


def fit_temperature(logits, y, lo=0.05, hi=20.0, iters=80):
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y)
    if len(y) < 20:
        return 1.0
    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(logits, y, np.exp(c)), nll(logits, y, np.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(logits, y, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(logits, y, np.exp(d))
    return float(np.exp((a + b) / 2))


def brier(probs, y):
    probs = np.asarray(probs, dtype=np.float64)
    onehot = np.eye(probs.shape[1])[np.asarray(y)]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def expected_calibration_error(probs, y, bins=15):
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y)
    conf, pred = probs.max(1), probs.argmax(1)
    hit = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - hit[m].mean())
    return float(ece)


def risk_coverage(probs, y, thresholds=None):
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y)
    conf, pred = probs.max(1), probs.argmax(1)
    if thresholds is None:
        thresholds = np.linspace(1.0 / probs.shape[1], 0.99, 50)
    rows = []
    for t in thresholds:
        m = conf >= t
        rows.append({"threshold": float(t), "coverage": float(m.mean()),
                     "accuracy": float((pred[m] == y[m]).mean()) if m.any() else float("nan")})
    return rows


def select_threshold(probs, y, target_accuracy=0.95, min_coverage=0.30, default=0.65):
    rows = risk_coverage(probs, y, np.linspace(0.30, 0.99, 70))
    for r in rows:
        if r["coverage"] >= min_coverage and r["accuracy"] >= target_accuracy:
            return r["threshold"], {"reached_target": True, **r}
    return default, {"reached_target": False, "threshold": default,
                     "note": "target accuracy not reachable on validation data; using the default threshold"}


class NoveltyDetector:

    def __init__(self, shrinkage=0.15):
        self.shrinkage = shrinkage
        self.means = None
        self.precision = None
        self.dim = 0
        self.threshold = None

    def fit(self, emb, y, n_cls):
        emb = np.asarray(emb, dtype=np.float64)
        y = np.asarray(y)
        d = emb.shape[1]
        self.dim = d
        present = [k for k in range(n_cls) if (y == k).sum() >= 2]
        if len(present) < 2 or len(emb) < d + 2:
            self.means = None
            return self
        means = np.full((n_cls, d), np.nan)
        centred = []
        for k in present:
            e = emb[y == k]
            means[k] = e.mean(0)
            centred.append(e - means[k])
        centred = np.concatenate(centred)
        cov = centred.T @ centred / max(len(centred) - 1, 1)
        cov = (1 - self.shrinkage) * cov + self.shrinkage * (np.trace(cov) / d) * np.eye(d)
        self.means = means
        self.precision = np.linalg.pinv(cov + 1e-6 * np.eye(d))
        return self

    @property
    def ready(self):
        return self.means is not None and self.threshold is not None

    def score(self, emb):
        if self.means is None:
            return np.zeros(len(np.atleast_2d(emb)))
        e = np.atleast_2d(np.asarray(emb, dtype=np.float64))
        best = np.full(len(e), np.inf)
        for mu in self.means:
            if np.isnan(mu).any():
                continue
            diff = e - mu
            best = np.minimum(best, np.einsum("ij,jk,ik->i", diff, self.precision, diff))
        return best / self.dim

    def calibrate(self, val_emb, quantile=0.995, slack=1.5):
        if self.means is None or len(val_emb) < 10:
            self.threshold = None
            return None
        s = self.score(val_emb)
        self.threshold = float(slack * max(np.quantile(s, quantile), 1.0))
        return self.threshold

    def is_novel(self, emb):
        if not self.ready:
            return np.zeros(len(np.atleast_2d(emb)), dtype=bool)
        return self.score(emb) > self.threshold

    def to_dict(self):
        if self.means is None:
            return None
        return {"means": np.nan_to_num(self.means, nan=0.0).tolist(),
                "valid": [bool(not np.isnan(m).any()) for m in self.means],
                "precision": self.precision.tolist(), "dim": self.dim, "threshold": self.threshold}

    @classmethod
    def from_dict(cls, d):
        o = cls()
        if not d:
            return o
        o.means = np.array(d["means"], dtype=np.float64)
        for k, ok in enumerate(d["valid"]):
            if not ok:
                o.means[k] = np.nan
        o.precision = np.array(d["precision"], dtype=np.float64)
        o.dim = int(d["dim"])
        o.threshold = d["threshold"]
        return o


def macro_f1_from(y, pred, n_cls):
    f1 = []
    for k in range(n_cls):
        tp = np.sum((y == k) & (pred == k))
        fp = np.sum((y != k) & (pred == k))
        fn = np.sum((y == k) & (pred != k))
        if tp + fn == 0:
            continue
        p, r = tp / max(tp + fp, 1), tp / (tp + fn)
        f1.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1)) if f1 else 0.0


def block_bootstrap_ci(y, pred, groups, n_cls, n_boot=1000, alpha=0.05, seed=0):
    y, pred, groups = np.asarray(y), np.asarray(pred), np.asarray(groups)
    ids = np.unique(groups)
    if len(ids) < 5:
        return None
    rng = np.random.default_rng(seed)
    idx_of = {g: np.flatnonzero(groups == g) for g in ids}
    accs, f1s = [], []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([idx_of[g] for g in pick])
        accs.append(float(np.mean(y[idx] == pred[idx])))
        f1s.append(macro_f1_from(y[idx], pred[idx], n_cls))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"recordings": int(len(ids)),
            "accuracy": [float(v) for v in np.percentile(accs, q)],
            "macro_f1": [float(v) for v in np.percentile(f1s, q)]}


def window_noise_floor(windows):
    w = np.asarray(windows, dtype=np.float32)
    if w.ndim != 3 or len(w) == 0:
        return 0.0
    return float(np.median(w.std(axis=1).mean(axis=1)))


class RoomAdapter:

    def __init__(self, train_floor, lo=0.33, hi=3.0):
        self.train_floor = float(train_floor)
        self.lo, self.hi = lo, hi

    def gain(self, room_floor):
        if self.train_floor <= 0 or room_floor <= 0:
            return 1.0
        return float(np.clip(self.train_floor / room_floor, self.lo, self.hi))


class HysteresisDecider:

    def __init__(self, k=4, switch_frames=2):
        self.k, self.switch_frames = k, switch_frames
        self.hist = []
        self.current = None
        self._pending, self._count = None, 0

    def reset(self):
        self.hist, self.current, self._pending, self._count = [], None, None, 0

    def update(self, probs):
        self.hist.append(np.log(np.asarray(probs, dtype=np.float64) + EPS))
        self.hist = self.hist[-self.k:]
        avg = softmax(np.mean(self.hist, axis=0))
        top = int(avg.argmax())
        if self.current is None:
            self.current = top
        elif top != self.current:
            if self._pending == top:
                self._count += 1
            else:
                self._pending, self._count = top, 1
            if self._count >= self.switch_frames:
                self.current, self._pending, self._count = top, None, 0
        else:
            self._pending, self._count = None, 0
        return self.current, avg
