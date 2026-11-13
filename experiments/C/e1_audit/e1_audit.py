"""(C) E1 audit: do existing Laplace-Fourier operators recover the true poles of an LTI system?

Plan: docs/proposal_C_identifiability.qmd, RQ1 / E1.

Data. Fixed pole-residue systems (``SYSTEMS``), window T = 20, n_t = 1024. Inputs are
an equal mix of step, ramp, PRBS and smooth random signals (non-periodic on [0, T]);
outputs are exact (``lfo.data.lti.response``, zero initial state). 200 train /
50 validation / 100 test samples, data seed 0. With the true poles loaded, a
single pole-residue layer reproduces the test outputs to ~1e-3 ("oracle" error).

Settings.
  layer  one pole-residue layer, one channel, no lifting/projection; the learned
         poles are directly comparable with the truth. n_poles = number of true
         poles (conjugates counted). One layer of LNO v2 equals ALNO here, so v2
         has no separate entry.
           tied   LNO v1 layer: transient + steady from the same poles
           alno   transient + free Fourier multiplier on the lowest ``modes``
           lfno   transient + R * (pole steady coefficients) on the lowest ``modes``
  deep   the full architectures (LNO1d port, ALNO1d, LFNO1d), parameter budget matched
         to the LFNO paper's ODE configuration. All poles of all layers and channels
         are pooled; the reported pole error is that of the closest pooled poles
         (Hungarian matching), i.e. an optimistic bound on recovery.

Protocols.
  common      ComplexAdam, lr 0.002, wd 1e-4, StepLR 100/0.5, batch 20, 1000 epochs
              (original LNO 1D protocol)
  lfno_paper  torch Adam, lr 0.002, wd 0.01, StepLR 100/0.8, batch 64, 1000 epochs
              (LFNO workshop paper, Appendix C); LFNO only.

Diagnostic inits (layer setting only, ``layer_init``): ``band`` (Im mu within the steady
band) and ``warm`` (truth + 20 % noise). They separate optimisation failure from
non-identifiability: published inits answer "what do the models learn as used";
``warm`` answers "is the truth a stable point of training".

Control (``--pole-lr-mult m``): the poles get learning rate m * lr, everything else
unchanged. Under Adam the total pole travel is bounded by about sum(lr * steps)
(4.0 for ``common``); m = 10 raises it to 40. Runs are labelled ``<protocol>+plr<m>``.

Pole conventions. Outputs are Re(sum gamma_n e^{mu_n t}), so one learned complex pole
can represent a conjugate pair. Before matching, learned and true poles are folded
to Im >= 0 and the true set is reduced to one pole per pair. For the residue error
(layer setting) the reference residue of a folded complex pair is 2 * beta (one
learned pole carrying the whole pair); a learned pole folded from Im < 0 has its
residue conjugated. Metrics come from ``lfo.metrics`` unchanged.

Usage:
    python experiments/C/e1_audit/e1_audit.py --setting layer --system p2 p4 p6 \
        --model tied alno lfno --init lno stable --seeds 0 1 2 3 4
    python experiments/C/e1_audit/e1_audit.py --summarize

Outputs: one JSON per run in experiments/C/e1_audit/results/<setting>/ (committed),
checkpoints in runs/C/e1_audit/ (git-ignored).
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lfo.data import lti
from lfo.metrics import pole_error, relative_l2, residue_error, seed_spread
from lfo.models import ALNO1d, LFNO1d, LNO1d
from lfo.models.hybrid import PoleResidueTS1d, SpectralConv1d
from lfo.training import ComplexAdam, count_params, relative_l2_loss, seed_everything

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
RUNS = ROOT / "runs" / "C" / "e1_audit"

T_END, N_T = 20.0, 1024
N_TRAIN, N_VALI, N_TEST = 200, 50, 100
MODES = 16                       # Fourier modes of the steady branch (ALNO, LFNO), as in the LFNO paper

# name -> (poles, residues); conjugate-symmetric, all stable
SYSTEMS = {
    "p2": ([-0.5 + 2j, -0.5 - 2j], [0.5 - 0.3j, 0.5 + 0.3j]),
    "p4": ([-0.4 + 3j, -0.4 - 3j, -1.0, -0.2], [0.3 - 0.4j, 0.3 + 0.4j, 1.0, -0.5]),
    "p6": ([-0.5 + 1.5j, -0.5 - 1.5j, -0.3 + 4j, -0.3 - 4j, -1.0, -2.0],
           [0.5 - 0.3j, 0.5 + 0.3j, 0.4 + 0.2j, 0.4 - 0.2j, 1.0, -0.8]),
}

PROTOCOLS = {
    "common": dict(optimizer="complex_adam", lr=0.002, weight_decay=1e-4, step_size=100, gamma=0.5,
                   batch_size=20, epochs=1000),
    "lfno_paper": dict(optimizer="adam", lr=0.002, weight_decay=0.01, step_size=100, gamma=0.8,
                       batch_size=64, epochs=1000),
}


# ---- data -----------------------------------------------------------------------

def time_grid() -> np.ndarray:
    return np.linspace(0, T_END, N_T + 1)[:-1]


def make_inputs(t: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Equal mix of the four input families, random parameters, shuffled."""
    fams = [
        lambda: lti.step(t, rng.uniform(0, T_END / 2), rng.uniform(0.5, 2.0) * rng.choice([-1, 1])),
        lambda: lti.ramp(t, rng.uniform(0, T_END / 2), rng.uniform(0.05, 0.3) * rng.choice([-1, 1])),
        lambda: lti.prbs(t, rng, hold=rng.uniform(0.3, 2.0)),
        lambda: lti.smooth_random(t, rng),
    ]
    u = np.stack([fams[i % 4]() for i in range(n)])
    return u[rng.permutation(n)]


def make_data(system: str, data_seed: int = 0):
    mu, beta = (np.array(a, dtype=complex) for a in SYSTEMS[system])
    t = time_grid()
    rng = np.random.default_rng(data_seed)
    u = make_inputs(t, N_TRAIN + N_VALI + N_TEST, rng)
    y = lti.response(t, u, mu, beta)
    f = lambda a: torch.tensor(a, dtype=torch.float32)
    s = np.cumsum([0, N_TRAIN, N_VALI, N_TEST])
    splits = {k: (f(u[a:b]), f(y[a:b])) for k, a, b in zip(("train", "vali", "test"), s[:-1], s[1:])}
    return splits, t, mu, beta


# ---- models ---------------------------------------------------------------------

class Layer1(nn.Module):
    """One single-channel pole-residue layer on (batch, n_t, 1); no lifting or projection."""

    def __init__(self, kind: str, n_poles: int, t, pole_init: str, modes: int = MODES):
        super().__init__()
        self.kind = kind
        self.pr = PoleResidueTS1d(1, 1, n_poles, t, pole_init)
        if kind in ("alno", "lfno"):
            self.spec = SpectralConv1d(1, 1, modes)
        elif kind != "tied":
            raise ValueError(kind)
        self.modes = modes

    def branches(self, x):
        """(transient, steady) outputs, each (b, 1, n_t)."""
        n = x.size(-1)
        tr = self.pr.transient(x)
        if self.kind == "tied":
            st = torch.fft.irfft(self.pr.steady_coeffs(x), n=n)
        elif self.kind == "alno":
            st = self.spec(x)
        else:
            st = torch.fft.irfft(self.spec.mix(self.pr.steady_coeffs(x, self.modes)), n=n)
        return tr, st

    def forward(self, x):
        tr, st = self.branches(x.permute(0, 2, 1))
        return (tr + st).permute(0, 2, 1)

    def pole_sets(self):
        mu, beta = self.pr.poles()
        return mu.flatten(), beta.flatten()


def _deep_model(kind: str, width: int, k: int, t, pole_init: str):
    """k = number of modes (LNO1d: poles; ALNO: poles per layer, FNO modes fixed; LFNO: modes = poles)."""
    if kind == "lno_v1":
        return LNO1d(width, k, t, use_grid=False, activation="sin")
    if kind == "alno":
        return ALNO1d(width, k, MODES, t, pole_init=pole_init)
    if kind == "lfno":
        return LFNO1d(width, k, t, pole_init=pole_init)
    raise ValueError(kind)


def deep_config(kind: str, t) -> dict:
    """Width / mode count whose parameter count is closest to the LFNO paper configuration."""
    target = count_params(LFNO1d(4, 16, t))
    if kind == "lfno":
        return dict(width=4, k=16, params=target, target=target)
    best = None
    for w, k in itertools.product(range(2, 13), (4, 8, 12, 16, 24, 32)):
        p = count_params(_deep_model(kind, w, k, t, "stable"))
        if best is None or abs(p - target) < abs(best["params"] - target):
            best = dict(width=w, k=k, params=p, target=target)
    return best


def deep_poles(model):
    if isinstance(model, LNO1d):
        mus, betas = [model.conv0.weights_pole.detach()], [model.conv0.weights_residue.detach()]
    else:
        mus, betas = zip(*model.poles())
    return torch.cat([m.flatten() for m in mus]), torch.cat([b.flatten() for b in betas])


# ---- pole conventions -------------------------------------------------------------

def fold(mu, beta):
    """Fold poles to Im >= 0 (residues conjugated where a pole was flipped)."""
    mu, beta = np.asarray(mu, complex), np.asarray(beta, complex)
    flip = mu.imag < 0
    return np.where(flip, mu.conj(), mu), np.where(flip, beta.conj(), beta)


def true_folded(mu, beta):
    """One pole per conjugate pair (Im >= 0); reference residue 2*beta for complex pairs."""
    keep = mu.imag >= 0
    m, b = mu[keep], beta[keep]
    return m, np.where(m.imag > 0, 2 * b, b)


@torch.no_grad()
def layer_init(model: "Layer1", init: str, mu_true, beta_true, rng: np.random.Generator):
    """E1-only diagnostic initialisations of the single layer (not part of any published model).

    band  Re mu = -exp(U[log 1/T, log 10/T]) as in "stable", but Im mu ~ U[0, omega_MODES]
          (the steady band), instead of up to the Nyquist/4.
    warm  the truth perturbed by 20 %: Re and Im parts each scaled by (1 + 0.2 z), z ~ N(0, 1)
          independently, so a stable pole stays stable (a complex factor on mu would rotate
          lightly damped poles into Re > 0); same for beta. Needs n_poles = number of true
          poles. Tests local convergence to the truth.
    """
    n = model.pr.mu.numel()
    if init == "band":
        re = -np.exp(rng.uniform(np.log(1 / T_END), np.log(10 / T_END), n))
        im = rng.uniform(0, 2 * np.pi * MODES / T_END, n)
        model.pr.mu.copy_(torch.tensor(re + 1j * im, dtype=torch.cfloat).reshape(1, 1, -1))
    elif init == "warm":
        if n != mu_true.size:
            raise ValueError("warm init needs n_poles = number of true poles")
        jitter = lambda z: z.real * (1 + 0.2 * rng.normal(size=n)) + 1j * z.imag * (1 + 0.2 * rng.normal(size=n))
        model.pr.mu.copy_(torch.tensor(jitter(mu_true), dtype=torch.cfloat).reshape(1, 1, -1))
        model.pr.beta.copy_(torch.tensor(jitter(beta_true), dtype=torch.cfloat).reshape(1, 1, -1))


# ---- training -------------------------------------------------------------------

POLE_PARAMS = ("pr.mu", ".mu", "weights_pole")


def param_groups(model, lr: float, pole_lr_mult: float):
    """Poles (``*.mu``, LNO port ``weights_pole``) get lr * pole_lr_mult; everything else lr."""
    poles, rest = [], []
    for name, p in model.named_parameters():
        (poles if name.endswith(POLE_PARAMS) else rest).append(p)
    return [dict(params=poles, lr=lr * pole_lr_mult), dict(params=rest, lr=lr)]


def train(model, data, proto: dict, device, pole_lr_mult: float = 1.0, penalty=None, hook=None):
    """penalty(model, epoch) -> scalar added to the loss; hook(model, epoch) after each epoch
    (both optional, used by the E2 remedies; E1 runs use neither)."""
    (xtr, ytr), (xva, yva) = data["train"], data["vali"]
    xtr, ytr, xva, yva = (a.to(device)[..., None] for a in (xtr, ytr, xva, yva))
    if proto["optimizer"] == "complex_adam":
        opt = ComplexAdam(param_groups(model, proto["lr"], pole_lr_mult), lr=proto["lr"],
                          weight_decay=proto["weight_decay"])
    else:
        opt = torch.optim.Adam(param_groups(model, proto["lr"], pole_lr_mult), lr=proto["lr"],
                               weight_decay=proto["weight_decay"])
    sch = torch.optim.lr_scheduler.StepLR(opt, proto["step_size"], proto["gamma"])
    bs, n = proto["batch_size"], xtr.shape[0]
    hist = []
    for ep in range(proto["epochs"]):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = relative_l2_loss(model(xtr[idx]), ytr[idx])
            (loss + penalty(model, ep) if penalty is not None else loss).backward()
            opt.step()
            tot += loss.item() * idx.numel()
        sch.step()
        if hook is not None:
            hook(model, ep)
        if (ep + 1) % 100 == 0 or ep == 0:
            hist.append((ep + 1, tot / n, evaluate(model, xva, yva)))
    return hist


@torch.no_grad()
def evaluate(model, x, y) -> float:
    model.eval()
    p = model(x)
    return float(np.mean(relative_l2(p[..., 0], y[..., 0], axis=1)))


@torch.no_grad()
def oracle_error(data, t, mu, beta) -> float:
    layer = PoleResidueTS1d(1, 1, mu.size, t)
    layer.mu.copy_(torch.tensor(mu, dtype=torch.cfloat).reshape(1, 1, -1))
    layer.beta.copy_(torch.tensor(beta, dtype=torch.cfloat).reshape(1, 1, -1))
    x, y = data["test"]
    return float(np.mean(relative_l2(layer(x[:, None])[:, 0], y, axis=1)))


# ---- one run --------------------------------------------------------------------

def protocol_label(protocol: str, pole_lr_mult: float) -> str:
    return protocol if pole_lr_mult == 1 else f"{protocol}+plr{pole_lr_mult:g}"


def run(setting, system, kind, init, protocol, seed, device, n_poles=None, pole_lr_mult=1.0):
    data, t, mu_true, beta_true = make_data(system)
    seed_everything(seed)
    if setting == "layer":
        n = n_poles or mu_true.size
        model = Layer1(kind, n, t, init if init in ("lno", "stable") else "stable")
        if init in ("band", "warm"):
            layer_init(model, init, mu_true, beta_true, np.random.default_rng(seed))
        cfg = dict(n_poles=n, modes=MODES)
    else:
        cfg = deep_config(kind, t)
        model = _deep_model(kind, cfg["width"], cfg["k"], t, init)
    model.to(device)
    proto = PROTOCOLS[protocol]
    t0 = time.time()
    hist = train(model, data, proto, device, pole_lr_mult)
    secs = time.time() - t0
    xte, yte = (a.to(device)[..., None] for a in data["test"])
    test = evaluate(model, xte, yte)

    mu_l, beta_l = model.pole_sets() if setting == "layer" else deep_poles(model)
    mu_l, beta_l = mu_l.cpu().numpy(), beta_l.cpu().numpy()
    mt, bt = true_folded(mu_true, beta_true)
    diverged = not (all(torch.isfinite(v).all() for v in model.state_dict().values()) and np.isfinite(test))
    ok = np.isfinite(mu_l) & np.isfinite(beta_l)   # a diverged run is scored on its finite poles only
    mf, bf = fold(mu_l[ok], beta_l[ok])
    if mf.size == 0:
        mf, bf = np.full(1, 1e30 + 0j), np.zeros(1, complex)   # no usable pole: error 1e30 relative, flagged diverged
    res = dict(
        setting=setting, system=system, model=kind, init=init, seed=seed,
        protocol=protocol_label(protocol, pole_lr_mult), base_protocol=protocol, pole_lr_mult=pole_lr_mult,
        config=cfg, params=count_params(model), seconds=round(secs, 1),
        test_rel_l2=test, diverged=bool(diverged), oracle_rel_l2=oracle_error(data, t, mu_true, beta_true),
        history=hist,
        true_poles=[[float(z.real), float(z.imag)] for z in mt],
        pole_error=pole_error(mf, mt, reduce="none").tolist(),
        pole_error_mean=pole_error(mf, mt),
        unstable_fraction=float(np.mean(mu_l[ok].real > 0)) if ok.any() else float("nan"),
        n_learned_poles=int(mu_l.size),
    )
    if setting == "layer":
        res["residue_error"] = residue_error(mf, bf, mt, bt, reduce="none").tolist()
        res["learned_poles"] = [[float(z.real), float(z.imag)] for z in mu_l]
        res["learned_residues"] = [[float(z.real), float(z.imag)] for z in beta_l]
        with torch.no_grad():
            tr, st = model.branches(xte[..., 0][:, None])
            out = tr + st
            res["transient_energy_fraction"] = float((tr.norm() ** 2 / (tr.norm() ** 2 + st.norm() ** 2)).item())
            res["branch_cancellation"] = float((out.norm() / (tr.norm() + st.norm())).item())
    else:
        # closest pooled pole to each true pole (after folding)
        from lfo.metrics import match_poles
        m = match_poles(mf, mt)
        closest = np.full(mt.size, np.nan, complex)
        closest[m.true_idx] = mf[m.pred_idx]
        res["closest_poles"] = [[float(z.real), float(z.imag)] for z in closest]

    if diverged:                # pole metrics of a diverged run are not meaningful
        for k in ("pole_error", "pole_error_mean", "residue_error"):
            if k in res:
                res[k] = None
    name = f"{system}_{kind}_{init}_{protocol_label(protocol, pole_lr_mult)}_s{seed}"
    if n_poles:
        name += f"_n{n_poles}"
    out_dir = RESULTS / setting
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1))
    RUNS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RUNS / f"{setting}_{name}.pt")
    print(f"{setting} {name}: params {res['params']} test {test:.3e} pole_err {res['pole_error_mean'] if diverged else f"{res['pole_error_mean']:.3f}"} "
          f"unstable {res['unstable_fraction']:.2f} ({secs:.0f}s)", flush=True)
    return res


# ---- summary --------------------------------------------------------------------

def summarize():
    lines = ["# E1 audit summary", "",
             "mean ± std over seeds. pole error: relative, after folding and Hungarian matching "
             "(deep: closest pooled poles). spread: mean over true poles of `seed_spread`. "
             "Diverged runs (non-finite weights) are counted under seeds and excluded from the statistics.", ""]
    for setting in ("layer", "deep"):
        files = sorted((RESULTS / setting).glob("*.json"))
        if not files:
            continue
        rs = [json.loads(f.read_text()) for f in files]
        groups = {}
        for r in rs:
            n = r["config"].get("n_poles")
            key = (r["system"], r["model"], r["init"], r["protocol"], n if setting == "layer" else None)
            groups.setdefault(key, []).append(r)
        lines += [f"## {setting}", "",
                  "| system | model | init | protocol | n | params | seeds | test rel. L2 | pole error | "
                  "seed spread | unstable | transient frac. |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
            g = groups[key]
            nd = sum(bool(r.get("diverged")) for r in g)
            g_ok = [r for r in g if not r.get("diverged")] or g
            te = np.array([r["test_rel_l2"] for r in g_ok])
            pe = np.array([r["pole_error_mean"] for r in g_ok])
            un = np.array([r["unstable_fraction"] for r in g_ok])
            mt = np.array([complex(*z) for z in g[0]["true_poles"]])
            if setting == "layer":
                sets = [fold(np.array([complex(*z) for z in r["learned_poles"]]),
                             np.zeros(len(r["learned_poles"])))[0] for r in g_ok]
                tf = np.array([r["transient_energy_fraction"] for r in g_ok])
                tfs = f"{tf.mean():.2f} ± {tf.std():.2f}"
            else:
                sets = [np.array([complex(*z) for z in r["closest_poles"]]) for r in g_ok]
                tfs = "–"
            sp = float(np.mean(seed_spread(sets, mt)))
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {key[4] or '–'} | {g[0]['params']} | {len(g)}{f' ({nd} div.)' if nd else ''} | "
                         f"{te.mean():.2e} ± {te.std():.1e} | {pe.mean():.3f} ± {pe.std():.3f} | {sp:.3f} | "
                         f"{un.mean():.2f} | {tfs} |")
        lines.append("")
    oracle = {}
    for f in (RESULTS / "layer").glob("*.json"):
        r = json.loads(f.read_text())
        oracle[r["system"]] = r["oracle_rel_l2"]
    if oracle:
        lines += ["Oracle (true poles in one layer), test rel. L2: " +
                  ", ".join(f"{k} {v:.1e}" for k, v in sorted(oracle.items())), ""]
    (RESULTS / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["layer", "deep"], default="layer")
    ap.add_argument("--system", nargs="+", default=list(SYSTEMS))
    ap.add_argument("--model", nargs="+", default=None)
    ap.add_argument("--init", nargs="+", default=["lno", "stable"],
                    help="lno, stable; layer setting also band, warm (E1 diagnostics)")
    ap.add_argument("--protocol", nargs="+", default=["common"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-poles", type=int, default=None, help="layer setting: override the number of learned poles")
    ap.add_argument("--pole-lr-mult", type=float, default=1.0,
                    help="learning-rate multiplier for the poles only (E1 control: pole travel budget)")
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = a.model or (["tied", "alno", "lfno"] if a.setting == "layer" else ["lno_v1", "alno", "lfno"])
    if a.epochs:                # smoke test: separate, git-ignored output folders
        global RESULTS, RUNS
        RESULTS, RUNS = RUNS / "_smoke", RUNS / "_smoke"
        for p in PROTOCOLS.values():
            p["epochs"] = a.epochs
    for system, kind, init, protocol, seed in itertools.product(a.system, models, a.init, a.protocol, a.seeds):
        if protocol == "lfno_paper" and kind != "lfno":
            continue
        if a.setting == "deep" and init in ("band", "warm"):
            raise SystemExit("band/warm inits are defined for the layer setting only")
        if kind == "lno_v1" and init != "lno":
            continue            # the LNO port has only the original initialisation
        run(a.setting, system, kind, init, protocol, seed, device, a.n_poles, a.pole_lr_mult)


if __name__ == "__main__":
    main()
