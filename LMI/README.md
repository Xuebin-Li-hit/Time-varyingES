# LMI feasibility for Proposition 1

| script | what it does |
|---|---|
| `check_lmi.py` | solves the three LMIs of Proposition 1 as an SDP and reports a feasibility margin, for the example of Section VI-A and for sweeps over `gamma` and `v` |
| `param_search_sim.py` | simulates the `d = 3` example of Fig. 2 for several parameter sets |

Requires `cvxpy` (CLARABEL or SCS).

## Status: every figure in the paper is strictly feasible

| figure | `1/(q rho)` | `gamma` | `alpha*k` | `m, M` | margin |
|---|---|---|---|---|---|
| Fig. 2 — §VI-A, constant-frequency, `d = 3` | 0.2 (`beta/v`) | 0.1 | 0.5 | 1.75, 4 | **+0.0103** |
| Fig. 3 — §VI-B 1, chirpy, `d = 1` | 0.2 | 0.05 | 0.6 | 1.75, 4 | **+0.0130** |
| Fig. 4 — §VI-B 2, chirpy, `d = 1` | 0.2 | 0.05 | 1.6 | 2, 2 | **+0.3756** |

All three land on `1/(q rho) = 0.2`. Figs. 2 and 3 use the cost family
`||x-c||^2 + ln(1+||x-c||^2)`, whose `M = 4` makes the `delta*M^2 = 16*delta`
term of `Phi_11` tight, hence the small margins; Fig. 4 uses pure quadratics
(`m = M = 2`) and is far more relaxed.

The parameters originally submitted — `alpha = k = gamma = 1`, `v = 2`,
`beta = 1` for §VI-A and `lambda = 0.03` for §VI-B — gave a margin of −0.201,
so the figures illustrated cases outside the hypotheses of Theorems 1 and 2.
Reviewer #4 was already probing this (Comment 5 asks under what conditions the
matrices of Proposition 1 exist).

Beyond the LMIs, Theorem 2 also needs `c - p < -2`. For the time-invariant
examples `c -> -infinity` and this is automatic, but Fig. 4's targets have
bounded non-decaying derivatives, so `c = 0` and `p > 2` is required. The
submitted `q = 3, v = 2` gave `p = q - v - 1 = 0` and violated it; `q = 4`
gives `p = 2.5, 3, 4`.

## A bug the feasibility work uncovered

Both chirpy scripts computed the probing phase as `cos(omega_t * t + ...)` where
`omega_t = omega * rho * (phi^q - 1)`. The phase in (14) is `omega * tau` with
`tau = t_0 + rho (phi^q - 1)`, i.e. `omega_t` itself — the extra `* t` makes the
argument `rad*s` and destroys the chirp demodulation. Fixed in
`fig4_chirpy_invariant.py` and `fig5_chirpy_varying.py`. Fig. 2 is unaffected:
its probing is constant-frequency, so `omega_s * t` is the correct phase there.

Related: once `phi` is saturated, the warped time `tau` must keep advancing at
the frozen rate `d(tau)/dt = phi_cap^(p+1)`. Freezing `tau` along with `phi`
stops the probing oscillation altogether and the algorithm dies.
