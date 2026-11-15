import torch
import torch.nn as nn
import torch.nn.functional as F


class ActivityNet(nn.Module):

    def __init__(self, n_feat=64, n_cls=4, drop=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_feat, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(drop * 0.3),
            nn.Conv1d(64, 96, 5, padding=2), nn.BatchNorm1d(96), nn.ReLU(), nn.Dropout(drop * 0.3),
        )
        self.temporal = nn.LSTM(96, 64, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(128, 32), nn.Tanh(), nn.Linear(32, 1))
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, n_cls))

    def forward(self, x):
        h = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.temporal(h)
        a = torch.softmax(self.attention(h), dim=1)
        return self.head((a * h).sum(dim=1))

    def features(self, x):
        logits = self.forward(x)
        return logits, logits


class _SqueezeExcite(nn.Module):

    def __init__(self, c, r=4):
        super().__init__()
        self.fc1 = nn.Linear(c, max(c // r, 4))
        self.fc2 = nn.Linear(max(c // r, 4), c)

    def forward(self, x):
        w = torch.sigmoid(self.fc2(F.gelu(self.fc1(x.mean(dim=2)))))
        return x * w.unsqueeze(2)


class _ResBlock(nn.Module):
    def __init__(self, cin, cout, k=5, dilation=1, drop=0.1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.c1 = nn.Conv1d(cin, cout, k, padding=pad, dilation=dilation, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=pad, dilation=dilation, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.se = _SqueezeExcite(cout)
        self.drop = nn.Dropout(drop)
        self.skip = nn.Conv1d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.drop(F.gelu(self.b1(self.c1(x))))
        h = self.se(self.b2(self.c2(h)))
        return F.gelu(h + self.skip(x))


class ActivityNetV2(nn.Module):
    EMB = 64

    def __init__(self, n_feat=64, n_cls=4, drop=0.3, width=96):
        super().__init__()
        self.n_feat, self.n_cls = n_feat, n_cls
        cin = 2 * n_feat
        branch = width // 3
        self.stem = nn.ModuleList([nn.Conv1d(cin, branch, k, padding=k // 2, bias=False) for k in (3, 5, 9)])
        self.stem_bn = nn.BatchNorm1d(3 * branch)
        self.block1 = _ResBlock(3 * branch, width, 5, 1, drop * 0.3)
        self.block2 = _ResBlock(width, width + 32, 5, 2, drop * 0.3)
        self.temporal = nn.LSTM(width + 32, 64, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(128, 32), nn.Tanh(), nn.Linear(32, 1))
        self.embed = nn.Sequential(nn.Linear(3 * 128, self.EMB), nn.LayerNorm(self.EMB), nn.GELU())
        self.drop = nn.Dropout(drop)
        self.classifier = nn.Linear(self.EMB, n_cls)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for name, p in self.temporal.named_parameters():
            if "weight_hh" in name:
                for i in range(0, p.shape[0], p.shape[0] // 4):
                    nn.init.orthogonal_(p.data[i:i + p.shape[0] // 4])
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(p.data)
            elif "bias" in name:
                nn.init.zeros_(p.data)
                n = p.shape[0] // 4
                p.data[n:2 * n] = 1.0
        for blk in (self.block1, self.block2):
            nn.init.zeros_(blk.b2.weight)

    def features(self, x):
        x = x.transpose(1, 2)
        dx = F.pad(x[:, :, 1:] - x[:, :, :-1], (1, 0))
        z = torch.cat([x, dx], dim=1)
        h = torch.cat([conv(z) for conv in self.stem], dim=1)
        h = F.gelu(self.stem_bn(h))
        h = self.block2(self.block1(h))
        h, _ = self.temporal(h.transpose(1, 2))
        a = torch.softmax(self.attention(h), dim=1)
        pooled = torch.cat([(a * h).sum(1), h.mean(1), h.max(1).values], dim=1)
        emb = self.embed(pooled)
        return self.classifier(self.drop(emb)), emb

    def forward(self, x):
        return self.features(x)[0]


ARCHS = {"v1": ActivityNet, "v2": ActivityNetV2}


def build_model(arch, n_feat, n_cls):
    if arch not in ARCHS:
        raise ValueError(f"unknown architecture '{arch}'")
    return ARCHS[arch](n_feat, n_cls)
