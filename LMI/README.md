# LMI feasibility for Proposition 1

| script | what it does |
|---|---|
| `check_lmi.py` | solves the three LMIs of Proposition 1 as an SDP and reports a feasibility margin, for the example of Section VI-A and for sweeps over `gamma` and `v` |
| `param_search_sim.py` | simulates the `d = 3` example of Fig. 2 for several parameter sets |

Requires `cvxpy` (CLARABEL or SCS).

## Status: every figure in the paper is strictly feasible

| figure | `1/(q rho)` | `gamma` | `alpha*k` | `m, M` | margin | certificate |
|---|---|---|---|---|---|---|
| Fig. 2 — §VI-A, constant-frequency, `d = 3` | 0.1 (`beta/v`) | 0.1 | 0.4 | 1.75, 4 | **+0.0126** | `p11=2.325, p22=1, delta=0.0719` |
| Fig. 3 — §VI-B 1, chirpy, `d = 1` | 0.1 | 0.05 | 0.4 | 1.75, 4 | **+0.0140** | `p11=2.375, p22=1, delta=0.0733` |
| Fig. 4 — §VI-B 2, chirpy, `d = 1` | 0.2 | 0.05 | 1.6 | 2, 2 | **+0.2435** | `p11=2.511, p22=1, delta=1.697` |

Figs. 2 and 3 use the cost family `||x-c||^2 + ln(1+||x-c||^2)`, whose `M = 4`
makes the `delta*M^2 = 16*delta` term of `Phi_11` tight, hence the small
margins; Fig. 4 uses pure quadratics (`m = M = 2`) and is far more relaxed —
which is why it alone survived the sign correction below without retuning.

The parameters originally submitted — `alpha = k = gamma = 1`, `v = 2`,
`beta = 1` for §VI-A and `lambda = 0.03` for §VI-B — gave a margin of −0.201,
so the figures illustrated cases outside the hypotheses of Theorems 1 and 2.
Reviewer #4 was already probing this (Comment 5 asks under what conditions the
matrices of Proposition 1 exist).

## The sign of the (1,3) block of `Phi_2`

The manuscript printed `Phi_2[1,3] = +(p22-p11) alpha k I/2`, but the appendix's
own eq. (25) gives the coefficient of `u_{f2:N}' g~` as `-(p22-p11) alpha k`.
Working `Vdot_g` out from `V = zeta_1' P zeta_1` and the dynamics (20),

    Vdot_g = -alpha k [ p11 u_f1' r' g + p22 u' g~ + w' P2' g~ ],

and substituting the orthogonality identity `xbar_f' g = u_f1' r' g + u' g~`
(from `xbar_f = T ubar_f`, `T = [r, R]` orthogonal) yields exactly eq. (25).
Eq. (26) flipped that sign when transcribing, and `Phi_2` inherited it.

This is not cosmetic: `diag(I, I, -I)` flips the (1,3) *and* (2,3) blocks
together, so flipping (1,3) alone is not a congruence and does change
definiteness. With the correct sign the first gain set loses its certificate,

| figure | printed `+` | corrected `-` |
|---|---|---|
| Fig. 2 at `beta/v=0.2, gamma=0.1, alpha*k=0.5` | +0.0103 | **−0.0169** |
| Fig. 3 at `1/(q rho)=0.2, gamma=0.05, alpha*k=0.6` | +0.0130 | **−0.0249** |
| Fig. 4 at `1/(q rho)=0.2, gamma=0.05, alpha*k=1.6` | +0.3756 | +0.2435 |

and the whole `m = 1.75, M = 4` family is infeasible on the entire row
`1/(q rho) = 0.2`, for every `gamma` in `[0.005, 1]` and `alpha*k` in
`[0.24, 4]`. Halving both `1/(q rho)` and `alpha*k` restores it, which is what
the table at the top now records. Halving `1/(q rho)` doubles every time
constant, so the horizons of Figs. 2 and 3 were doubled to match and the panels
keep their shape.

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
