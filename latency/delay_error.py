"""Delay-error model and hand-off-aware planning (side paper, the proposed method).

Post hoc, for a FROZEN world model and a delay d, two small networks are fitted on
the model's own d-step prediction residuals on D_train windows
(r = z(t+d) - rollout(z(t), a(t:t+d))):

  error model   (z_hat, z(t), a(t:t+d)) -> mean and log-variance of r
                used when a call starts: the post-delay anchor is z_hat + mean,
                and hedge anchors are drawn from N(z_hat + mean, var).
  hand-off head (z, a(0:d)) -> log-variance of r
                the predicted uncertainty of the NEXT call's anchor if the actions
                a(0:d) are committed from state z; used to steer commitments.

Planning arms built from them (latency/run_dmc.py):

  ecpp<d>k<K>         error-model hedge: K anchors (corrected mean first)
  ecpp<d>k<K>h<lam>   + hand-off penalty: lam x std_P(J) x z-score of the
                      candidate's predicted next-hand-off log-variance
  ecpp<d>k<K>r<lam>   + generic risk penalty: lam x std over anchors of J
                      (mean-std robust objective; the baseline the hand-off
                      penalty must beat)
  mcpp<d>             mean correction only (one corrected anchor)
  scpp<d>k<K>         homoscedastic hedge: anchors from N(z_hat + mean, s^2)
                      with s the global residual standard deviation
Models are cached in results/latency/error_model/models/.
"""
import pathlib
import re

import numpy as np
import torch
from torch import nn

from experiments.dmc import evaluate as EV
from paths import RESULTS

CTX = 16
CACHE = RESULTS / "latency" / "error_model" / "models"
ARM = re.compile(r"(ecpp|mcpp|scpp)(\d+)(?:k(\d+))?(?:([hr])([\d.]+))?")


class Net(nn.Module):
    def __init__(self, n_in, D, hid=512, mean=True):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(n_in, hid), nn.SiLU(), nn.Linear(hid, hid), nn.SiLU())
        self.lv = nn.Linear(hid, D)
        self.mu = nn.Linear(hid, D) if mean else None
        if mean:
            nn.init.zeros_(self.mu.weight); nn.init.zeros_(self.mu.bias)

    def forward(self, x):
        h = self.f(x)
        mu = self.mu(h) if self.mu is not None else torch.zeros_like(self.lv(h))
        return mu, self.lv(h).clamp(-12, 6)


def _fit(X, R, dev, mean=True, steps=3000, bs=256, seed=0):
    torch.manual_seed(seed)
    xm, xs = X.mean(0), X.std(0) + 1e-6
    rm, rs = (R.mean(0), R.std(0) + 1e-6) if mean else (np.zeros(R.shape[1]), R.std(0) + 1e-6)
    Xt = torch.as_tensor((X - xm) / xs, dtype=torch.float32, device=dev)
    Rt = torch.as_tensor((R - rm) / rs, dtype=torch.float32, device=dev)
    n = len(Xt); nv = n // 10
    perm = torch.randperm(n, device=dev); va, tr = perm[:nv], perm[nv:]
    net = Net(Xt.shape[1], Rt.shape[1], mean=mean).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = 1e9, None
    for s in range(steps):
        i = tr[torch.randint(0, len(tr), (bs,), device=dev)]
        mu, lv = net(Xt[i])
        loss = (0.5 * (lv + (Rt[i] - mu) ** 2 / lv.exp())).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 99:
            with torch.no_grad():
                mu, lv = net(Xt[va]); vl = float((0.5 * (lv + (Rt[va] - mu) ** 2 / lv.exp())).mean())
            if vl < best:
                best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    norm = {k: torch.as_tensor(v, dtype=torch.float32, device=dev) for k, v in
            dict(xm=xm, xs=xs, rm=rm, rs=rs).items()}
    return net, norm, best


class DelayError:
    """Error model + hand-off head for one (world model, delay)."""

    def __init__(self, em, em_norm, ho, ho_norm, s_global, d):
        self.em, self.en, self.ho, self.hn, self.d = em, em_norm, ho, ho_norm, d
        self.s_global = s_global

    @torch.no_grad()
    def predict(self, zh, zt, prefix):
        """torch (n, D), (n, D), (n, d, A) -> mean (n, D), std (n, D) of the residual."""
        x = torch.cat([zh, zt, prefix.reshape(len(prefix), -1)], 1).float()
        mu, lv = self.em((x - self.en["xm"]) / self.en["xs"])
        return mu * self.en["rs"] + self.en["rm"], torch.exp(0.5 * lv) * self.en["rs"]

    @torch.no_grad()
    def handoff_logvar(self, z, acts):
        """torch (n, D), (n, d, A) -> (n,) mean log-variance of the next hand-off residual."""
        x = torch.cat([z, acts.reshape(len(acts), -1)], 1).float()
        _, lv = self.ho((x - self.hn["xm"]) / self.hn["xs"])
        return (lv + 2 * torch.log(self.hn["rs"])).mean(1)


def fit_delay_error(ad, meta, d, dev, n_train=4096, seed=5, H=32):
    """Fit (or load) the error model and hand-off head for one model and delay."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{meta['family']}_{meta['task']}_{meta['run']}_d{d}.pt"
    if f.exists():
        ck = torch.load(f, map_location=dev, weights_only=False)
        em = Net(ck["em_in"], ck["D"]).to(dev); em.load_state_dict(ck["em"]); em.eval()
        ho = Net(ck["ho_in"], ck["D"], mean=False).to(dev); ho.load_state_dict(ck["ho"]); ho.eval()
        nz = lambda n: {k: v.to(dev) for k, v in n.items()}
        return DelayError(em, nz(ck["en"]), ho, nz(ck["hn"]), ck["s_global"].to(dev), d)
    td = EV.task_data(meta["task"])
    rng = np.random.default_rng(seed + d)
    e = rng.integers(0, EV.N_TRAIN, n_train)
    t = rng.integers(CTX + 8, td.L - H - d - 2, n_train)
    Z, ACT = [], []
    for i in range(0, n_train, 128):
        ei, ti = e[i:i + 128], t[i:i + 128]
        obs = np.stack([td.frames[a, b - CTX:b + d + 1] for a, b in zip(ei, ti)])
        act = np.stack([td.action[a, b - CTX:b + d] for a, b in zip(ei, ti)])
        Z.append(np.asarray(ad.encode_sequence(obs, act), np.float64)[:, CTX:]); ACT.append(act[:, CTX:])
    Z, ACT = np.concatenate(Z), np.concatenate(ACT)
    zh = np.asarray(ad.rollout(Z[:, 0], ACT[:, :d]), np.float64)[:, -1]
    R = Z[:, d] - zh
    pre = ACT[:, :d].reshape(n_train, -1)
    em, en, _ = _fit(np.concatenate([zh, Z[:, 0], pre], 1), R, dev, mean=True)
    ho, hn, _ = _fit(np.concatenate([Z[:, 0], pre], 1), R, dev, mean=False)
    s_global = torch.as_tensor(R.std(0), dtype=torch.float32, device=dev)
    torch.save({"em": em.state_dict(), "ho": ho.state_dict(), "en": {k: v.cpu() for k, v in en.items()},
                "hn": {k: v.cpu() for k, v in hn.items()}, "em_in": 2 * Z.shape[-1] + pre.shape[1],
                "ho_in": Z.shape[-1] + pre.shape[1], "D": Z.shape[-1], "s_global": s_global.cpu(),
                "n_train": n_train}, f)
    return DelayError(em, en, ho, hn, s_global, d)


def make_cost_delay_error(ad, P, dev, amp, de, kind, K, pen, lam, H, gen_seed=5):
    """make_cost(Z) for the cpp schedule with prefix d = de.d (see the module docstring)."""
    d = de.d
    gen = torch.Generator(device=dev).manual_seed(gen_seed)
    ac = (lambda: torch.autocast("cuda", dtype=torch.float16)) if amp else (lambda: torch.autocast("cpu", enabled=False))

    def make_cost(Z):
        Zt = torch.as_tensor(np.asarray(Z), dtype=torch.float32, device=dev)
        E = len(Zt)

        def cost(seq):
            L, A = seq.shape[1], seq.shape[2]
            if L == H:                                              # pre-episode plan: state known
                with ac():
                    return torch.as_tensor(ad.get_cost(Zt.repeat_interleave(P, 0), seq.float()),
                                           dtype=torch.float64, device=dev)
            s = seq.reshape(E, P, L, A)
            prefix = s[:, 0, :d].float()
            zh = torch.as_tensor(np.asarray(ad.rollout(Zt, prefix))[:, -1], dtype=torch.float32, device=dev)
            mu, sd = de.predict(zh, Zt, prefix)
            a0 = zh + mu
            free = s[:, :, d:].float().reshape(E * P, L - d, A)
            nk = 1 if kind == "mcpp" else K
            Js = []
            for k in range(nk):
                if k == 0:
                    anc = a0
                else:
                    eps = torch.randn(a0.shape, generator=gen, device=dev)
                    anc = a0 + eps * (de.s_global[None] if kind == "scpp" else sd)
                with ac():
                    Js.append(torch.as_tensor(ad.get_cost(anc.repeat_interleave(P, 0), free),
                                              dtype=torch.float64, device=dev).reshape(E, P))
            Js = torch.stack(Js, 0)                                     # (k, E, P)
            J = Js.mean(0)
            if lam and kind == "ecpp":
                sJ = J.std(1, keepdim=True).clamp_min(1e-8)
                if pen == "h":                                       # next hand-off predictability
                    u = de.handoff_logvar(a0.repeat_interleave(P, 0), s[:, :, d:2 * d].reshape(E * P, d, A).float())
                    u = u.double().reshape(E, P)
                    u = (u - u.mean(1, keepdim=True)) / u.std(1, keepdim=True).clamp_min(1e-8)
                    J = J + lam * sJ * u
                elif pen == "r":                                     # generic mean-std robustness
                    J = J + lam * Js.std(0)
            return J.reshape(-1)
        return cost
    return make_cost


def parse(arm):
    m = ARM.fullmatch(arm)
    if not m:
        return None
    kind, d, K, pen, lam = m.group(1), int(m.group(2)), int(m.group(3) or 1), m.group(4), float(m.group(5) or 0)
    return kind, d, K, pen, lam
