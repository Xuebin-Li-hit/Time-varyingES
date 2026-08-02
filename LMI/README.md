# LMI feasibility for Proposition 1

| script | what it does |
|---|---|
| `check_lmi.py` | solves the three LMIs of Proposition 1 as an SDP and reports a feasibility margin, for the example of Section VI-A and for sweeps over `gamma` and `v` |

Requires `cvxpy` (CLARABEL or SCS).

## Status: every figure in the paper is strictly feasible

| figure | `1/(q rho)` | `gamma` | `alpha*k` | `m, M` | margin | certificate |
|---|---|---|---|---|---|---|
| Fig. 2 — §VI-A, constant-frequency, `d = 3` | 0.1 (`beta/v`) | 0.1 | 0.4 | 1.75, 4 | **+0.0126** | `p11=2.325, p22=1, delta=0.0719` |
| Fig. 3 — §VI-B 1, chirpy, `d = 1` | 0.1 | 0.05 | 0.4 | 1.75, 4 | **+0.0140** | `p11=2.375, p22=1, delta=0.0733` |
| Fig. 4 — §VI-B 2, chirpy, `d = 1` | 0.2 / 0.1 | 0.05 | 1.6 | 2, 2 | **+0.2435 / +0.2820** | `p11=2.511, delta=1.697` and `p11=2.519, delta=1.819`, both at `p22=1` |

Fig. 4's asymptotic and exponential panels sit at `1/(q rho) = 0.2` and its
prescribed-time panel at `0.1`, hence the two certificates.

## What makes the margins what they are

**The cost family sets `m` and `M`, and `M` is the binding one.** Figs. 2 and 3
use `||x-c||^2 + ln(1+||x-c||^2)`, whose Hessian eigenvalues run over
`[1.75, 4]`. `Phi_11 = 2(beta/v) p11 - m alpha k p11 + delta M^2` therefore
carries `16*delta`, which is what keeps those two margins near `10^-2`. Fig. 4
uses pure quadratics, `m = M = 2`, and is far more relaxed.

**`Phi_11 < 0` forces `alpha k > 2 (beta/v) / m`, while `Phi_2` degrades as
`alpha k` grows** — its (1,3) and (2,3) blocks scale with it. The two
requirements squeeze `alpha k` from both sides, and for the `m = 1.75, M = 4`
family the usable window is roughly `[0.3, 0.6]` once `1/(q rho) = 0.1`.

**Lowering `1/(q rho)` widens that window.** For the same family, `0.2` admits
no `(gamma, alpha k)` at all, whereas `0.1` admits the row above and `0.05`
more. Since `1/(q rho)` also sets every time constant of the run, buying margin
this way costs proportionally longer horizons — which is why Figs. 2 and 3 are
plotted over 400 s and 60/24/10 s respectively.

**`v` is the wrong lever.** The LMIs see only the ratio `beta/v`, but the growth
of `xi(t) = (1 + beta t)^{1/v}` is governed mainly by the exponent `1/v`.
Keeping `v` at its smallest admissible value and lowering `beta` therefore buys
feasibility at the least cost in growth: `beta/v = 0.25` at `v = 2` gives
`xi(100) = 7.1`, the same ratio at `v = 10` only 1.5.

**`P3` must be symmetric.** `P` is the Lyapunov matrix of
`V = zeta_1' P zeta_1`, and the appendix computes `Vdot = 2 zeta_1' P zeta_1dot`,
which is valid only for symmetric `P`. A non-symmetric `P3` satisfies the matrix
inequality but certifies nothing, and `V` sees only its symmetric part, so the
relaxation buys nothing either. (The `P3'` appearing in `Phi_22` is transpose
bookkeeping for writing the (1,2) block rather than the (2,1) block.)

**The LMIs are homogeneous of degree one** in `(p11, p22, delta, P2, P3)`:
scaling a feasible point by any positive number keeps it feasible. A plain
feasibility problem therefore drifts to the origin, where every constraint holds
only to solver tolerance. `check_lmi.py` fixes the scale with `p22 = 1` (without
loss of generality, by homogeneity) and maximises a uniform margin instead, so
that a positive value certifies strict feasibility and a negative one proves
infeasibility.

## Beyond the LMIs

Theorem 2 also needs `c - p < -2`. For the time-invariant examples
`c -> -infinity` and this holds automatically. Fig. 4's targets have bounded but
non-decaying derivatives, so `c = 0` and `p > 2` is required, which is what
fixes `q = 4` for its asymptotic and exponential panels (`p = 2.5, 3`) and
`q = 2, varrho = 2` for its prescribed-time one (`p = 3`).

## Two implementation points the scripts depend on

**The chirp phase is `omega * tau`,** with `tau = t_0 + rho (phi^q - 1)`. It
carries no separate factor `t`; writing one would make the argument `rad*s` and
destroy the demodulation. Fig. 2 is constant-frequency, so `omega_s * t` is the
correct phase there.

**Once `phi` saturates, the warped time `tau` must keep advancing** at the
frozen rate `d(tau)/dt = phi_cap^(p+1)`. Freezing `tau` along with `phi` stops
the probing oscillation altogether and the algorithm dies.
