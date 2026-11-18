"""(C) E2 phase 2: does training recover the poles where phase 1 says the data identifies them?

Phase 1 (``e2_richness.py``) predicts, per data configuration, which poles are practically
identifiable (profiled-error ratio >= 10). Here the E1 single tied layer is trained on
exactly those data and the recovered poles are compared with the prediction.

Fixed across conditions (only the data change):
  model     E1 ``Layer1("tied")`` (LNO v1 layer), n_poles = number of true poles
  init      E1 ``band`` init with its T = 20 ranges (Re in [-0.5, -0.05], Im in [0, 5.03] rad/s),
            so the initial pole distribution is the same in rad/s for every window T
  protocol  E1 ``common`` with pole lr x 10 (``common+plr10``), 1000 epochs
Data: ``e2_richness.make`` (dt = 20/1024): train = the 200 samples analysed in phase 1
(data seed 0), validation 50 (seed 1), test 100 (seed 2), same family / K / T.

    python experiments/C/theory/e2_train.py --cond p4_e1_T20 p4_ms16_T20 --seeds 0 1 2
    python experiments/C/theory/e2_train.py --cond p4_ms16_T20 --variant real_min warm --seeds 0 1 2
    python experiments/C/theory/e2_train.py --summarize

Variants (controls for the real poles; ``base`` is the setting above):
  base        n_poles = number of true poles (conjugates counted), all slots complex, band init
  warm        as base, E1 ``warm`` init (truth, Re and Im each x (1 + 0.2 z))
  free_min    n_poles = number of folded true poles (one slot per pair or real pole), all complex
  real_spare  as base, but as many slots as there are real true poles have Im mu fixed to 0
  real_min    free_min with the real-pole slots constrained real: the minimal form of phase 1
Remedies (E2 -> E3/E4 candidates):
  wide        n_poles = (complex pairs + 1) + 2 * (real poles) slots, all band init (slot-count control)
  mixed       same slots as wide; (pairs + 1) band slots, 2 * (real poles) slots started on the real
              axis with Re = -geomspace(0.05, 5) * exp(0.2 z) (stratified, so they cannot start
              together) and constrained real (Im mask as in real_spare). Left free, such slots
              leave the axis at once: d/d(Im mu) = -t Im(beta) e^{Re mu t} != 0 for a complex
              residue, and Adam turns any small gradient into a step of about lr.
  mixed_off   mixed with the real grid shifted by half a log step (k = 4: 0.11, 0.50, 2.3, 10.7),
              a control: the unshifted grid (0.05, 0.23, 1.08, 5) lies close to p4's -0.2 and -1
  *_rep       real slots repel in log|Re|: + lambda(ep) sum_{i<j} exp(-(d_i - d_j)^2 / (2 sigma^2)),
              d = log|Re mu|, lambda = 1e-2 decaying linearly to 0 at epoch 600, sigma = 0.15
              (the true -1, -2 are 4.6 sigma apart)
  *_split     at epochs 150 and 300, each pair of real slots within 10 % is merged (residues added,
              output nearly unchanged) and the freed slot moved, with residue 0, to the point of
              the log grid on [-5, -0.05] farthest from all real slots
  *_l1, *_l1s  sparsity: + lambda(ep) sum_j |beta_j| ||e^{mu_j t}||_2 (impulse-response norm of each
              slot, so fast decays are not favoured), lambda = 1e-3 / 1e-2 to epoch 200, linearly
              to 0 at 300; at epoch 300 slots whose output is < 2 % of the layer output (train
              inputs) are pruned (residue masked to 0) and training continues to 1000 without
              penalty. Pole metrics use the remaining slots only.
  *_prune     pruning at epoch 300 without the penalty (control)
  few         mixed with n_real + 1 real slots (instead of 2 n_real) on a log grid of [-5, -0.05]
              with a random offset per seed (positions (i + u) / k, u ~ U(0, 1)), no shift prior
  fixed8      prior-free (no use of the true pole count): 8 slots, 4 band-init complex slots and
              4 real-constrained slots on a log grid of [-5, -0.05] with a random offset per seed
  gate8       as fixed8, but no slot is constrained: Im mu = sigmoid(a) Im mu~ with a learned gate
              logit a per slot (pole lr), a = -5 on the grid slots (start almost real: sigmoid = 0.007,
              Im mu~ from the band init, so |Im mu| <= 0.03) and +3 on the band slots; a slot becomes real if its gate closes
  bound8      as fixed8, but the complex slots are bounded away from the real axis:
              |Im mu| = 2 pi / T + |Im mu~| (at least one period in the window), so a slow decay
              can only be taken by a real slot; still no use of the true pole count
  damp8       as bound8, and the complex slots are also lightly damped: |Re mu| < c |Im mu| with
              c = 0.5 (Re mu = -c |Im mu| sigmoid(a), a learned), so a heavily damped oscillation
              cannot stand in for a fast real decay; damp8w: c = 1. True complex poles: |Re|/|Im| <= 0.33
  *_ms        multi-start: 8 candidate inits (seeds 100 * seed + c) trained 100 epochs each,
              the best on validation continues with the full protocol (1000 epochs, fresh optimiser)
Conditions ending in ``_foh``: the tied layer's response 1 / (i omega - mu) is replaced by the exact
sampled response to a first-order-hold (piecewise-linear) input, the input model of the data
generator (``FOHPoleResidue``); the oracle uses the same layer. E4 traced the layer's low-noise
"discretisation floor" to this hold mismatch (``e4_profile.py --hold foh``).
The real constraint is a parametrisation of ``pr.mu`` (Im multiplied by a 0/1 mask), applied
after the init, so the constrained slots start on the real axis at the band init's Re.

Outputs: results/e2_train/<cond>_s<seed>.json (committed), results/e2_train.md,
checkpoints in runs/C/e2/ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.utils.parametrize  # noqa: F401

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2_richness as R  # noqa: E402  (also puts e1_audit on the path)

E = R.E
from lfo.metrics import pole_error  # noqa: E402
from lfo.models.hybrid import PoleResidueTS1d  # noqa: E402
from lfo.training import count_params, seed_everything  # noqa: E402

OUT = R.RESULTS / "e2_train"
RUNS = E.ROOT / "runs" / "C" / "e2"
PROTOCOL, PLR = "common", 10.0
RECOVERED = 0.1
VARIANTS = ("base", "warm", "free_min", "real_spare", "real_min", "wide", "mixed", "base_ms", "mixed_ms", "mixed_off", "mixed_off_ms",
            "mixed_off_rep", "mixed_off_rep_ms", "mixed_off_split", "mixed_off_split_ms",
            "mixed_off_l1", "mixed_off_l1_ms", "mixed_off_l1s_ms", "mixed_off_prune_ms", "few", "few_ms",
            "fixed8", "fixed8_ms", "gate8", "gate8_ms", "bound8", "bound8_ms", "damp8", "damp8_ms", "damp8w", "damp8w_ms")
DAMP = {"damp8": 0.5, "damp8w": 1.0}                       # complex slots: |Re mu| <= c |Im mu|
N_GENERIC, GATE_ON, GATE_OFF = 8, 3.0, -5.0               # prior-free variants: 8 slots; gate logits
MS_CANDIDATES, MS_EPOCHS = 8, 100
REP_LAMBDA, REP_SIGMA, REP_EPOCHS = 1e-2, 0.15, 600       # repulsion weight (annealed to 0), width in log|Re|
SPLIT_EPOCHS, SPLIT_TOL = (150, 300), 0.1                 # coalescence check: |log(Re_i / Re_j)| < 0.1
GRID = np.log(np.geomspace(0.05, 5.0, 200))               # candidate log|Re| for a split slot
L1_LAMBDA = {"_l1": 1e-3, "_l1s": 1e-2}                    # sparsity weight, constant to epoch 200, 0 at 300
L1_ON, L1_OFF, PRUNE_EPOCH, PRUNE_TOL = 200, 300, 300, 0.02
E.POLE_PARAMS = E.POLE_PARAMS + ("mu.original", ".gate")   # parametrised pr.mu and real gates get the pole lr


class ImagGate(torch.nn.Module):
    """mu -> Re mu + i Im mu * sigmoid(gate): a learned, per-slot real/complex switch."""

    def __init__(self, logits):
        super().__init__()
        self.gate = torch.nn.Parameter(logits)

    def forward(self, mu):
        return torch.complex(mu.real, mu.imag * torch.sigmoid(self.gate))


class ImagFloor(torch.nn.Module):
    """Real slots (mask 0): Im mu = 0; complex slots: Im mu = w_min + |Im mu~|."""

    def __init__(self, mask, w_min):
        super().__init__()
        self.register_buffer("mask", mask)
        self.w_min = w_min

    def forward(self, mu):
        return torch.complex(mu.real, self.mask * (self.w_min + mu.imag.abs()))


class DampFloor(torch.nn.Module):
    """As ImagFloor, and complex slots are lightly damped: Re mu = -c Im mu sigmoid(Re mu~),
    so |Re mu| < c |Im mu| (damping ratio < c / sqrt(1 + c^2)); real slots: Re mu = Re mu~."""

    def __init__(self, mask, w_min, c):
        super().__init__()
        self.register_buffer("mask", mask)
        self.w_min, self.c = w_min, c

    def forward(self, mu):
        im = self.w_min + mu.imag.abs()
        re_c = -self.c * im * torch.sigmoid(mu.real)
        return torch.complex(self.mask * re_c + (1 - self.mask) * mu.real, self.mask * im)


class ImagMask(torch.nn.Module):
    """mu -> Re mu + i Im mu * mask: slots with mask 0 are real poles."""

    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, mu):
        return torch.complex(mu.real, mu.imag * self.mask)

class Bounded(torch.nn.Module):
    """Complex multiplier with modulus < delta: delta z / (1 + |z|)."""

    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self, z):
        return self.delta * z / (1 + torch.sqrt(z.real ** 2 + z.imag ** 2 + 1e-12))


class PartialTie(E.Layer1):
    """Steady branch tied to the poles over the full band, plus a bounded untied residual on
    the lowest ``modes`` rfft modes (H3: R = K + R_res):

      alno_pt   steady = irfft(K(x) + R_res x_hat)        R_res free per mode, |R_res| < delta
      lfno_pt   steady = irfft((1 + R_res) K(x))          LFNO's R = 1 + R_res, |R_res| < delta

    K(x) = the tied layer's steady coefficients. R_res starts at 0 and the truth has R_res = 0,
    so the oracle is the tied layer's. delta = 0 is the tied layer; delta -> inf on all modes
    approaches the untied branch (ALNO: R_res replaces K's role)."""

    def __init__(self, kind, n_poles, t, pole_init, modes, delta):
        super().__init__(kind.removesuffix("_pt"), n_poles, t, pole_init, modes)
        with torch.no_grad():
            self.spec.weight.zero_()
        torch.nn.utils.parametrize.register_parametrization(self.spec, "weight", Bounded(delta))

    def branches(self, x):
        n = x.size(-1)
        K = self.pr.steady_coeffs(x)
        res = self.spec.mix(torch.fft.rfft(x) if self.kind == "alno" else K)
        c = torch.cat([K[..., :self.modes] + res, K[..., self.modes:]], -1)
        return self.pr.transient(x), torch.fft.irfft(c, n=n)


# name -> (system, family, T, K)
CONDITIONS = {
    "p4_e1_T20": ("p4", "e1", 20.0, 0),            # E1 data: real poles predicted non-identifiable
    "p4_ramp_T20": ("p4", "ramp", 20.0, 0),        # negative control: nothing identifiable
    "p4_prbs_T20": ("p4", "prbs", 20.0, 0),
    "p4_ms16_T20": ("p4", "multisine", 20.0, 16),
    "p4_e1_T40": ("p4", "e1", 40.0, 0),
    "p4_ms16_T40": ("p4", "multisine", 40.0, 16),
    "p6_e1_T20": ("p6", "e1", 20.0, 0),
    # D1 third pass: E1 families with the truth on a 32x finer grid (no path exact by construction)
    "p6_e1fine_T20": ("p6", "e1fine", 20.0, 0),
    "p6_e1fine_T20_foh": ("p6", "e1fine", 20.0, 0, 1024, dict(hold="foh")),
    "p6_ms16_T80": ("p6", "multisine", 80.0, 16),  # richest p6 data: pairs identifiable, real poles not
    "p6_ms16_T80_N4096": ("p6", "multisine", 80.0, 16, 4096),  # dt = 20/4096: -1 identifiable (ratio 11.6)
    # E3: untied steady branch (layer setting), steady band 2 pi M / T = 20 rad/s (input band 0.1-10 rad/s)
    "p4_ms16_T20_alno": ("p4", "multisine", 20.0, 16, 1024, dict(kind="alno", modes=64)),
    "p4_ms16_T20_lfno": ("p4", "multisine", 20.0, 16, 1024, dict(kind="lfno", modes=64)),
    "p6_ms16_T80_N4096_alno": ("p6", "multisine", 80.0, 16, 4096, dict(kind="alno", modes=256)),
    "p6_ms16_T80_N4096_lfno": ("p6", "multisine", 80.0, 16, 4096, dict(kind="lfno", modes=256)),
    # H3: partial tie R = K + R_res, |R_res| < delta (absolute for ALNO, relative to K for LFNO)
    **{f"p6_ms16_T80_N4096_{k}_pt{tag}": ("p6", "multisine", 80.0, 16, 4096, dict(kind=f"{k}_pt", modes=256, delta=d))
       for k in ("alno", "lfno") for d, tag in ((0.03, "003"), (0.3, "03"))},
    **{f"p4_ms16_T20_{k}_pt03": ("p4", "multisine", 20.0, 16, 1024, dict(kind=f"{k}_pt", modes=64, delta=0.3))
       for k in ("alno", "lfno")},
    # E4: output noise on train and validation (test clean), relative L2 eta per sample
    "p4_ms16_T20_n1e-3": ("p4", "multisine", 20.0, 16, 1024, dict(noise=1e-3)),
    "p4_ms16_T20_n1e-2": ("p4", "multisine", 20.0, 16, 1024, dict(noise=1e-2)),
    "p6_ms16_T80_N4096_n1e-3": ("p6", "multisine", 80.0, 16, 4096, dict(noise=1e-3)),
    "p6_ms16_T80_N4096_n1e-2": ("p6", "multisine", 80.0, 16, 4096, dict(noise=1e-2)),
    "p4_ms16_T20_n1e-1": ("p4", "multisine", 20.0, 16, 1024, dict(noise=1e-1)),
    "p4_ms16_T20_n3e-1": ("p4", "multisine", 20.0, 16, 1024, dict(noise=3e-1)),
    "p6_ms16_T80_N4096_n1e-1": ("p6", "multisine", 80.0, 16, 4096, dict(noise=1e-1)),
    "p6_ms16_T80_N4096_n3e-1": ("p6", "multisine", 80.0, 16, 4096, dict(noise=3e-1)),
    # E4: tied layer with the first-order-hold response (the data generator's input model)
    "p4_ms16_T20_foh": ("p4", "multisine", 20.0, 16, 1024, dict(hold="foh")),
    "p6_ms16_T80_foh": ("p6", "multisine", 80.0, 16, 1024, dict(hold="foh")),
    "p6_ms16_T80_N4096_foh": ("p6", "multisine", 80.0, 16, 4096, dict(hold="foh")),
    "p6_ms16_T80_N4096_n1e-2_foh": ("p6", "multisine", 80.0, 16, 4096, dict(noise=1e-2, hold="foh")),
    "p6_ms16_T80_N4096_n1e-1_foh": ("p6", "multisine", 80.0, 16, 4096, dict(noise=1e-1, hold="foh")),
    "p4_e1_T20_foh": ("p4", "e1", 20.0, 0, 1024, dict(hold="foh")),       # E1 data with the FOH layer
    "p6_e1_T20_foh": ("p6", "e1", 20.0, 0, 1024, dict(hold="foh")),
    # E4: E1 data under output noise, around the CRB noise limit of the real poles (p6: -2 at 0.035, -1 at 0.11)
    **{f"{s}_e1_T20_n{e}": (s, "e1", 20.0, 0, 1024, dict(noise=float(e)))
       for s, es in (("p4", ("1e-1", "3e-1")), ("p6", ("1e-2", "3e-2", "1e-1"))) for e in es},
    # E4: input noise (errors-in-variables) on the train / validation inputs, under output noise 1e-2
    **{f"p4_ms16_T20_n1e-2_u{e}": ("p4", "multisine", 20.0, 16, 1024, dict(noise=1e-2, in_noise=float(e)))
       for e in ("1e-2", "3e-2", "1e-1", "3e-1")},
    **{f"p6_e1_T20_n1e-2_u{e}": ("p6", "e1", 20.0, 0, 1024, dict(noise=1e-2, in_noise=float(e)))
       for e in ("1e-2", "3e-2", "5e-2", "1e-1")},
    # E3 / E4: partial tie under noise (released from the noise-aware tied runs; no Adam runs of their own)
    **{f"p6_ms16_T80_N4096_{k}_pt03_n{e}": ("p6", "multisine", 80.0, 16, 4096,
                                            dict(kind=f"{k}_pt", modes=256, delta=0.3, noise=float(e)))
       for k in ("alno", "lfno") for e in ("1e-2", "1e-1")},
}
NOISE_GROUP = 5                                     # runs s, s' with s // 5 == s' // 5 share one noise draw
N_DEFAULT = 1024                                    # grid: dt = 20 / N (5th entry of a condition, optional)


def unpack(cond):
    c = CONDITIONS[cond]
    return (*c[:4], c[4] if len(c) > 4 else N_DEFAULT)


def extras(cond):
    """kind (tied / alno / lfno), steady modes, output noise level of a condition."""
    c = CONDITIONS[cond]
    return dict(dict(kind="tied", modes=E.MODES, noise=0.0, in_noise=0.0, delta=None, hold="band"), **(c[5] if len(c) > 5 else {}))


class FOHPoleResidue(PoleResidueTS1d):
    """PoleResidueTS1d with the sampled first-order-hold response instead of 1 / (i omega - mu):
    per pole the exact discrete filter x_{j+1} = E x_j + b0 u_j + b1 u_{j+1} of a piecewise-linear
    input, E = e^{mu dt}, I0 = (E - 1) / mu, I1 = (E - 1 - mu dt) / mu^2, b0 = I0 - I1 / dt,
    b1 = I1 / dt, gives H_l = (b0 + b1 z_l) / (z_l - E), z_l = e^{i omega_l dt}. The transient
    term keeps its form. Computed in complex128 (I1 cancels for small mu dt), returned in the
    layer's dtype."""

    def _H(self, iw):
        dt = float(self.t[1] - self.t[0])
        mu = self.mu.to(torch.complex128)
        z = mu * dt
        Ek = torch.exp(z)
        I0, I1 = (Ek - 1) / mu, (Ek - 1 - z) / mu ** 2
        zl = torch.exp(iw.to(torch.complex128) * dt)[:, None, None, None]
        H = ((I0 - I1 / dt) + (I1 / dt) * zl) / (zl - Ek)
        return (self.beta.to(torch.complex128) * H).to(self.beta.dtype)


def make_splits(system, family, T, K, N=N_DEFAULT, noise=0.0, noise_seed=0, in_noise=0.0):
    """noise > 0: white Gaussian noise on the train and validation outputs, per sample with
    relative L2 norm ``noise`` (std noise * ||y_i|| / sqrt(n_t)); the test outputs stay clean.
    in_noise > 0: the same on the train and validation inputs (errors-in-variables: the outputs
    come from the clean inputs); the test inputs stay clean."""
    R.DT = 20.0 / N
    f = lambda a: a.to(torch.float32)
    splits = {}
    for i, (name, seed, n) in enumerate((("train", 0, 200), ("vali", 1, 50), ("test", 2, 100))):
        t, u, y, _, _ = R.make(system, family, T, K, seed=seed)
        u, y = u[:n], y[:n]
        if noise > 0 and name != "test":
            g = torch.Generator().manual_seed(10_000 + 10 * noise_seed + i)
            y = y + noise * y.norm(dim=1, keepdim=True) / np.sqrt(y.shape[1]) * torch.randn(y.shape, generator=g,
                                                                                           dtype=y.dtype)
        if in_noise > 0 and name != "test":
            g = torch.Generator().manual_seed(20_000 + 10 * noise_seed + i)
            u = u + in_noise * u.norm(dim=1, keepdim=True) / np.sqrt(u.shape[1]) * torch.randn(u.shape, generator=g,
                                                                                              dtype=u.dtype)
        splits[name] = (f(u), f(y))
    return splits, t.numpy()


def setup(cond, seed):
    """Data, time grid and true system of a run (noise draw shared by groups of NOISE_GROUP seeds)."""
    system, family, T, K, N = unpack(cond)
    data, t = make_splits(system, family, T, K, N, extras(cond)["noise"], seed // NOISE_GROUP,
                          extras(cond)["in_noise"])
    mu_true, beta_true = (np.array(a, dtype=complex) for a in E.SYSTEMS[system])
    return data, t, mu_true, beta_true


def build_for(cond, variant, seed, t, mu_true, beta_true):
    x = extras(cond)
    return build(variant, mu_true, beta_true, t, seed, kind=x["kind"], modes=x["modes"], delta=x["delta"], hold=x["hold"])


@torch.no_grad()
def oracle_for(cond, data, t, mu_true, beta_true):
    """Test error of the true poles in this layer kind; untied steady branches set to their
    tied value on the kept modes (ALNO: weight = tied steady coefficient; LFNO: R = 1)."""
    x = extras(cond)
    if x["hold"] == "foh":
        layer = FOHPoleResidue(1, 1, mu_true.size, t)
        layer.mu.copy_(torch.tensor(mu_true, dtype=torch.cfloat).reshape(1, 1, -1))
        layer.beta.copy_(torch.tensor(beta_true, dtype=torch.cfloat).reshape(1, 1, -1))
        xte, yte = data["test"]
        return float(np.mean(E.relative_l2(layer(xte[:, None])[:, 0], yte, axis=1)))
    if x["kind"] == "tied" or x["kind"].endswith("_pt"):          # partial tie: truth has R_res = 0
        return E.oracle_error(data, t, mu_true, beta_true)
    m = E.Layer1(x["kind"], mu_true.size, t, "stable", x["modes"])
    m.pr.mu.copy_(torch.tensor(mu_true, dtype=torch.cfloat).reshape(1, 1, -1))
    m.pr.beta.copy_(torch.tensor(beta_true, dtype=torch.cfloat).reshape(1, 1, -1))
    if x["kind"] == "alno":
        imp = torch.zeros(1, 1, t.size)
        imp[..., 0] = 1.0                                   # rfft(impulse) = 1: steady coeffs = multiplier
        m.spec.weight.copy_(m.pr.steady_coeffs(imp, x["modes"])[0][None])
    else:
        m.spec.weight.copy_(torch.ones_like(m.spec.weight))
    xte, yte = (a[..., None] for a in data["test"])
    return E.evaluate(m, xte, yte)


def predicted(system, family, T, K, N=N_DEFAULT):
    """Phase-1 ratios for this configuration (Re mu, per true pole), if computed."""
    path = R.RESULTS / ("e2_richness.json" if N == N_DEFAULT else "e2_grid.json")
    if not path.exists():
        return None
    for r in json.loads(path.read_text()):
        if (r["system"], r["family"], r["T"], r["K"]) == (system, family, T, K) and r.get("n_per_20", N_DEFAULT) == N:
            return dict(floor=r["floor"], cond=r["cond"],
                        ratio={f"{p['pole'][0]:+g}{p['pole'][1]:+g}i": p["Re mu"]["ratio"] for p in r["poles"]})
    return None


def base_variant(variant):
    v = variant.removesuffix("_ms")
    for suf in ("_rep", "_split", "_l1s", "_l1", "_prune"):
        v = v.removesuffix(suf)
    return v


class Keep(torch.nn.Module):
    """beta -> beta * keep: pruned slots have keep 0."""

    def __init__(self, n):
        super().__init__()
        self.register_buffer("keep", torch.ones(1, 1, n))

    def forward(self, beta):
        return beta * self.keep


def keep_mask(model):
    if torch.nn.utils.parametrize.is_parametrized(model.pr, "beta"):
        return model.pr.parametrizations.beta[0].keep.flatten().bool()
    return torch.ones(model.pr.beta.numel(), dtype=torch.bool, device=model.pr.beta.device)


def make_l1(lam0):
    def l1(model, ep):
        lam = lam0 if ep < L1_ON else lam0 * max(0.0, (L1_OFF - ep) / (L1_OFF - L1_ON))
        if lam == 0:
            return 0.0
        mu, beta = model.pr.mu.flatten(), model.pr.beta.flatten()
        a = mu.real.clamp(max=-1e-4)
        T = float(model.pr.t[-1])
        w = torch.sqrt((1 - torch.exp(2 * a * T)) / (-2 * a))          # ||e^{mu t}||_2 on [0, T]
        return lam * (beta.abs() * w).sum()
    return l1


def make_pruner(events, x):
    @torch.no_grad()
    def prune(model, ep):
        if ep + 1 != PRUNE_EPOCH:
            return
        keep = model.pr.parametrizations.beta[0].keep
        full = model(x).norm()
        old = keep.clone()
        share = []
        for j in range(keep.numel()):
            if not old.flatten()[j]:
                share.append(0.0)
                continue
            keep.copy_(torch.zeros_like(old))
            keep.view(-1)[j] = 1
            share.append(float(model(x).norm() / full))
        keep.copy_(old)
        drop = [j for j, c in enumerate(share) if c < PRUNE_TOL]
        keep.view(-1)[drop] = 0
        mu = model.pr.mu.flatten().cpu().numpy()
        events.append(dict(epoch=ep + 1, share=share, pruned=drop,
                           pruned_poles=[[float(mu[j].real), float(mu[j].imag)] for j in drop]))
    return prune


def real_slots(model):
    """Indices of the real-constrained slots (Im mask 0), empty if none."""
    if not torch.nn.utils.parametrize.is_parametrized(model.pr, "mu"):
        return torch.empty(0, dtype=torch.long)
    return (model.pr.parametrizations.mu[0].mask.flatten() == 0).nonzero().flatten()


def repulsion(model, ep):
    lam = REP_LAMBDA * max(0.0, 1 - ep / REP_EPOCHS)
    idx = real_slots(model)
    if lam == 0 or idx.numel() < 2:
        return 0.0
    d = torch.log((-model.pr.mu.flatten()[idx].real).clamp_min(1e-3))
    diff = d[:, None] - d[None, :]
    iu = torch.triu_indices(d.numel(), d.numel(), 1)
    return lam * torch.exp(-diff[iu[0], iu[1]] ** 2 / (2 * REP_SIGMA ** 2)).sum()


def make_splitter(events):
    @torch.no_grad()
    def split(model, ep):
        if ep + 1 not in SPLIT_EPOCHS:
            return
        idx = real_slots(model).tolist()
        orig = model.pr.parametrizations.mu.original      # (1, 1, n); real slots: Im is masked anyway
        beta = model.pr.beta
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                ri, rj = -orig[0, 0, i].real, -orig[0, 0, j].real
                if ri > 0 and rj > 0 and abs(float(torch.log(ri / rj))) < SPLIT_TOL:
                    beta[0, 0, i] += beta[0, 0, j]
                    beta[0, 0, j] = 0
                    d = torch.log((-orig[0, 0, idx].real).clamp_min(1e-3)).cpu().numpy()
                    new = GRID[np.argmax(np.min(np.abs(GRID[:, None] - d[None, :]), axis=1))]
                    orig[0, 0, j] = torch.tensor(-np.exp(new), dtype=orig.dtype)
                    events.append(dict(epoch=ep + 1, merged=[i, j], at=float(-ri), moved_to=float(-np.exp(new))))
    return split


def build(variant, mu_true, beta_true, t, seed, kind="tied", modes=None, delta=None, hold="band"):
    full = variant
    variant = base_variant(variant)
    n_real = int(np.sum(mu_true.imag == 0))
    n_folded = int(np.sum(mu_true.imag >= 0))
    n_pairs = n_folded - n_real
    n = {"free_min": n_folded, "real_min": n_folded,
         "wide": n_pairs + 1 + 2 * n_real, "mixed": n_pairs + 1 + 2 * n_real,
         "mixed_off": n_pairs + 1 + 2 * n_real, "few": n_pairs + 1 + n_real + 1,
         "fixed8": N_GENERIC, "gate8": N_GENERIC, "bound8": N_GENERIC, "damp8": N_GENERIC, "damp8w": N_GENERIC}.get(variant, mu_true.size)
    model = (PartialTie(kind, n, t, "stable", modes, delta) if kind.endswith("_pt")
             else E.Layer1(kind, n, t, "stable", modes or E.MODES))
    if hold == "foh":
        assert kind == "tied" and type(model.pr) is PoleResidueTS1d
        model.pr.__class__ = FOHPoleResidue                 # before any parametrisation of pr.mu
    rng = np.random.default_rng(seed)
    E.layer_init(model, "warm" if variant == "warm" else "band", mu_true, beta_true, rng)
    if any(full.removesuffix("_ms").endswith(sfx) for sfx in ("_l1", "_l1s", "_prune")):
        torch.nn.utils.parametrize.register_parametrization(model.pr, "beta", Keep(n))
    if variant in ("fixed8", "gate8", "bound8", "damp8", "damp8w"):
        k = N_GENERIC // 2
        re = -np.exp(np.log(0.05) + (np.arange(k) + rng.uniform()) / k * np.log(100.0))
        with torch.no_grad():
            mu0 = model.pr.mu.clone()
            mu0[..., n - k:] = torch.complex(torch.tensor(re, dtype=torch.float32), mu0[..., n - k:].imag)
            model.pr.mu.copy_(mu0)
        if variant in ("fixed8", "bound8", "damp8", "damp8w"):
            mask = torch.ones(n)
            mask[n - k:] = 0
            if variant in DAMP:
                w_min, c = 2 * np.pi / (float(t[-1]) + float(t[1] - t[0])), DAMP[variant]
                with torch.no_grad():                   # band init's |Im| kept where possible, its Re
                    mu0 = model.pr.mu.clone()           # clipped into the cone and stored as a logit
                    im = mu0[..., :n - k].imag.abs().clamp_min(w_min)
                    q = (mu0[..., :n - k].real.abs() / (c * im)).clamp(0.02, 0.98)
                    mu0[..., :n - k] = torch.complex(torch.log(q / (1 - q)), im - w_min)
                    model.pr.mu.copy_(mu0)
                par = DampFloor(mask.reshape(1, 1, -1), w_min, c)
            elif variant == "bound8":
                w_min = 2 * np.pi / (float(t[-1]) + float(t[1] - t[0]))
                with torch.no_grad():                   # keep the band init's |Im| where possible
                    mu0 = model.pr.mu.clone()
                    mu0[..., :n - k] = torch.complex(mu0[..., :n - k].real, (mu0[..., :n - k].imag.abs() - w_min).clamp_min(0))
                    model.pr.mu.copy_(mu0)
                par = ImagFloor(mask.reshape(1, 1, -1), w_min)
            else:
                par = ImagMask(mask.reshape(1, 1, -1))
            torch.nn.utils.parametrize.register_parametrization(model.pr, "mu", par)
        else:
            logits = torch.full((n,), GATE_ON)
            logits[n - k:] = GATE_OFF
            torch.nn.utils.parametrize.register_parametrization(model.pr, "mu", ImagGate(logits.reshape(1, 1, -1)))
        return model, dict(n_poles=n, real_slots=k, gate=variant == "gate8")
    if variant in ("mixed", "mixed_off", "few"):
        k = n_real + 1 if variant == "few" else 2 * n_real
        if variant == "few":
            re = -np.exp(np.log(0.05) + (np.arange(k) + rng.uniform()) / k * np.log(100.0))
        else:
            shift = (5.0 / 0.05) ** (0.5 / (k - 1)) if variant == "mixed_off" else 1.0   # half a grid step
            re = -np.geomspace(0.05, 5.0, k) * shift * np.exp(0.2 * rng.normal(size=k))
        with torch.no_grad():
            model.pr.mu[..., n - k:] = torch.tensor(re, dtype=torch.cfloat)
        mask = torch.ones(n)
        mask[n - k:] = 0
        torch.nn.utils.parametrize.register_parametrization(model.pr, "mu", ImagMask(mask.reshape(1, 1, -1)))
        return model, dict(n_poles=n, real_slots=k)
    if variant in ("real_spare", "real_min"):
        mask = torch.ones(n)
        mask[n - n_real:] = 0
        torch.nn.utils.parametrize.register_parametrization(model.pr, "mu", ImagMask(mask.reshape(1, 1, -1)))
    return model, dict(n_poles=n, real_slots=n_real if variant.startswith("real") else 0)


def remedy(variant, events, x=None):
    v = variant.removesuffix("_ms")
    for suf, lam in L1_LAMBDA.items():
        if v.endswith(suf):
            return dict(penalty=make_l1(lam), hook=make_pruner(events, x))
    if v.endswith("_prune"):
        return dict(hook=make_pruner(events, x))
    if v.endswith("_rep"):
        return dict(penalty=repulsion)
    if v.endswith("_split"):
        return dict(hook=make_splitter(events))
    return {}


def run_name(cond, variant, seed):
    return f"{cond}_s{seed}" if variant == "base" else f"{cond}_{variant}_s{seed}"


def run(cond: str, seed: int, device, variant: str = "base"):
    system, family, T, K, N = unpack(cond)
    data, t, mu_true, beta_true = setup(cond, seed)
    t0 = time.time()
    xprune = data["train"][0][:50].to(device)[..., None]
    if variant.endswith("_ms"):
        screen = dict(E.PROTOCOLS[PROTOCOL], epochs=MS_EPOCHS)
        cands = []
        for c in range(MS_CANDIDATES):
            seed_everything(100 * seed + c)
            m, cfg = build_for(cond, variant, 100 * seed + c, t, mu_true, beta_true)
            m.to(device)
            h = E.train(m, data, screen, device, PLR, **remedy(variant, [], xprune))
            cands.append((h[-1][2] if np.isfinite(h[-1][2]) else np.inf, c, m))
        val, best, model = min(cands, key=lambda z: z[0])
        cfg = dict(cfg, ms_candidates=MS_CANDIDATES, ms_epochs=MS_EPOCHS, ms_best=best,
                   ms_vals=sorted(float(z[0]) for z in cands))
        seed_everything(seed)
    else:
        seed_everything(seed)
        model, cfg = build_for(cond, variant, seed, t, mu_true, beta_true)
        model.to(device)
    events = []
    hist = E.train(model, data, E.PROTOCOLS[PROTOCOL], device, PLR, **remedy(variant, events, xprune))
    if events:
        cfg = dict(cfg, events=events)
    secs = time.time() - t0
    xte, yte = (a.to(device)[..., None] for a in data["test"])
    test = E.evaluate(model, xte, yte)
    active = keep_mask(model).cpu().numpy()
    mu_l, beta_l = (a.detach().cpu().numpy()[active] for a in model.pole_sets())
    mt, _ = E.true_folded(mu_true, beta_true)
    diverged = not (np.isfinite(mu_l).all() and np.isfinite(test))
    mf, _ = E.fold(mu_l, beta_l)
    pe = None if diverged else pole_error(mf, mt, reduce="none").tolist()
    res = dict(cond=cond, variant=variant, config=cfg, system=system, family=family, T=T, K=K, seed=seed, n_t=int(t.size),
               model=extras(cond)["kind"], hold=extras(cond)["hold"], modes=extras(cond)["modes"], noise=extras(cond)["noise"], in_noise=extras(cond)["in_noise"], delta=extras(cond)["delta"], init="warm" if variant == "warm" else "band", protocol=f"{PROTOCOL}+plr{PLR:g}", params=count_params(model),
               seconds=round(secs, 1), test_rel_l2=test, diverged=bool(diverged), history=hist,
               oracle_rel_l2=oracle_for(cond, data, t, mu_true, beta_true),
               true_poles=[[float(z.real), float(z.imag)] for z in mt], pole_error=pe,
               pole_error_mean=None if pe is None else float(np.mean(pe)),
               learned_poles=[[float(z.real), float(z.imag)] for z in mu_l],
               gates=(torch.sigmoid(model.pr.parametrizations.mu[0].gate).flatten().detach().cpu().numpy()[active].tolist()
                      if cfg.get("gate") else None),
               predicted=predicted(system, family, T, K, N), n_per_20=N)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{run_name(cond, variant, seed)}.json").write_text(json.dumps(res, indent=1))
    RUNS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RUNS / f"{run_name(cond, variant, seed)}.pt")
    print(f"{cond} {variant} s{seed}: test {test:.2e} pole_err {pe and [round(x, 3) for x in pe]} ({secs:.0f}s)", flush=True)


def summarize():
    rs = [json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))]
    for r in rs:
        r.setdefault("variant", "base")
    lines = ["# E2 phase 2: training on the phase-1 data configurations", "",
             "Tied layer, band init, `common` + pole lr × 10, 1000 epochs. Per true pole: phase-1 ratio "
             f"(Re μ; **bold** ≥ {R.RATIO_OK:g} = predicted identifiable) / fraction of seeds recovering it "
             f"(relative pole error < {RECOVERED}) / mean ± std pole error. Diverged runs excluded.", ""]
    for system in ("p4", "p6"):
        conds = [(c, v) for c in CONDITIONS for v in VARIANTS
                 if CONDITIONS[c][0] == system and any(r["cond"] == c and r["variant"] == v for r in rs)]
        if not conds:
            continue
        poles = next(r["true_poles"] for r in rs if r["system"] == system)
        labels = [f"{z[0]:+g}{z[1]:+g}i" if z[1] else f"{z[0]:+g}" for z in poles]
        keys = [f"{z[0]:+g}{z[1]:+g}i" for z in poles]
        lines += [f"## {system}", "",
                  "| condition | variant | slots | n_t | seeds | test rel. L2 | oracle | " + " | ".join(labels) + " |",
                  "|---|---|---|---|---|---|---|" + "---|" * len(poles)]
        for c, v in conds:
            g = [r for r in rs if r["cond"] == c and r["variant"] == v]
            ok = [r for r in g if not r["diverged"]]
            te = np.array([r["test_rel_l2"] for r in ok])
            pe = np.array([r["pole_error"] for r in ok])
            pred = g[0]["predicted"] or {"ratio": {}}
            cells = []
            for j, k in enumerate(keys):
                q = pred["ratio"].get(k, float("nan"))
                qs = f"**{q:.1f}**" if q >= R.RATIO_OK else f"{q:.1f}"
                cells.append(f"{qs} / {np.mean(pe[:, j] < RECOVERED):.0%} / {pe[:, j].mean():.2f} ± {pe[:, j].std():.2f}")
            div = f" ({len(g) - len(ok)} div.)" if len(g) > len(ok) else ""
            slots = len(g[0]["learned_poles"])
            lines.append(f"| {c} | {v} | {slots} | {g[0]['n_t']} | {len(g)}{div} | {te.mean():.1e} ± {te.std():.1e} | "
                         f"{g[0]['oracle_rel_l2']:.1e} | " + " | ".join(cells) + " |")
        lines.append("")
    (R.RESULTS / "e2_train.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--variant", nargs="+", default=["base"], choices=VARIANTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for c in a.cond:
        for v in a.variant:
            for s in a.seeds:
                run(c, s, device, v)


if __name__ == "__main__":
    main()
