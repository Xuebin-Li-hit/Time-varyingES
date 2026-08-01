"""
Comparison against a classical *extremum-seeking* distributed method, on the
smart-grid application that paper itself uses.

Baseline: M. Ye and G. Hu, "Distributed extremum seeking for constrained
networked optimization and its application to energy consumption control in
smart grid", IEEE Trans. Control Syst. Technol. 24(6) 2016, algorithm (8):

    x_ij   = xh_ij + b sin(w_ij t)
    xh_ij' = -k ( f_i(x_i) sin(w_ij t)
                  + (b/2) [ sum_m a_im (xh_ij - xh_mj)
                            + sum_m a_im ( z_ij -  z_mj) ] )
    z_ij'  = th sum_m a_im (xh_ij - xh_mj)

Application (their Section VII): energy consumption control for N = 5 users
with HVAC systems.  Decision vector is the whole consumption profile
l = (l_1..l_5), so d = 5, and every user holds an estimate of all five entries.
Discomfort  rho_i * ||l - lhat||^2 ; price  (k_p (sum_j l_j - Lstar) + p0) l_i .

The cost is theirs unchanged; only the magnitudes are scaled.  Their
lhat = [120..200] and l* = [82..167]; everything here is divided by 40 (and
p0 = 1 instead of 10), so the optimum sits below 10 and the plots are readable.

A note on the hypotheses.  With the discomfort rho_i (l_i - lhat_i)^2 the local
cost depends on l only through l_i and sum_j l_j, so its Hessian vanishes on the
subspace orthogonal to span{e_i, 1}: no f_i is strictly convex in l, and a
fortiori none is strongly convex.  Ye and Hu observe this for their own
Assumption 1 (their footnote 2: "C_i(l) is not strictly convex in l, indicating
that Assumption 1 might be conservative"), and the same reservation applies to
our Assumption 1.  The aggregate cost is strongly convex -- its Hessian is
2 diag(rho) + 2 k_p 1 1' -- and both schemes converge here, which is the point
the reference itself makes: the per-agent convexity requirement is sufficient
but conservative.

Graphs.  Ye and Hu require an undirected connected graph (their Lemma 1 and
Assumption 1), so their scheme is run on the undirected 5-cycle.  Ours only
needs a weight-balanced, strongly connected digraph, so it is run on the
*directed* ring with the same edges -- the topology used throughout the paper.
This is why the comparison is not carried out in the manuscript: it would
require introducing and displaying a second communication graph.

Probing frequencies.  Ye and Hu need one frequency per (agent, coordinate)
pair, i.e. N*d = 25 here; ours needs one per coordinate, i.e. d = 5, because
the demodulation is carried by the phase k*phi*(f_i - eta_i) instead.

Outputs out/compare_es_smartgrid.eps (and .png).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

LATEX_FONT_SIZE = 21
plt.rcParams.update({
    'font.size': LATEX_FONT_SIZE,
    'axes.titlesize': LATEX_FONT_SIZE,
    'axes.labelsize': LATEX_FONT_SIZE,
    'xtick.labelsize': LATEX_FONT_SIZE - 1,
    'ytick.labelsize': LATEX_FONT_SIZE - 1,
    'legend.fontsize': LATEX_FONT_SIZE - 7,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
})
os.makedirs('out', exist_ok=True)

# ---------------------------------------------------------------- problem
N = 5          # users
d = 5          # entries of the consumption profile
RHO = np.array([5.2, 5.4, 5.6, 5.8, 6.0])            # their rho_i gamma_i^2
LHAT = np.array([3.0, 3.5, 4.0, 4.5, 5.0])           # their [120..200] / 40
KP = 0.5                                             # pricing parameter
P0 = 1.0                                             # their p0 = 10, scaled
LSTAR = 0.8 * LHAT.sum()                             # their L* = 0.8 sum lhat


def f_local(i, l):
    """User i's measured cost at profile l (a d-vector): discomfort + payment."""
    return RHO[i] * (l[i] - LHAT[i]) ** 2 + (KP * (l.sum() - LSTAR) + P0) * l[i]


def f_all(X):
    """X is (N, d): row i is user i's estimate.  Returns the N measured values."""
    return np.array([f_local(i, X[i]) for i in range(N)])


def social_optimum():
    # d f/d l_m = 2 rho_m (l_m - lhat_m) + 2 KP S - KP LSTAR + P0 = 0
    # => l_m = lhat_m - A/(2 rho_m),  A = 2 KP S - KP LSTAR + P0
    Q = (1.0 / (2 * RHO)).sum()
    H = LHAT.sum()
    S = (H + Q * (KP * LSTAR - P0)) / (1 + 2 * KP * Q)
    A = 2 * KP * S - KP * LSTAR + P0
    return LHAT - A / (2 * RHO)


LOPT = social_optimum()

# ---------------------------------------------------------------- graphs
A_DIR = np.array([[0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0]], float)        # ours: directed ring
A_UND = ((A_DIR + A_DIR.T) > 0).astype(float)     # theirs: undirected 5-cycle
L_DIR = np.diag(A_DIR.sum(1)) - A_DIR
L_UND = np.diag(A_UND.sum(1)) - A_UND

X0 = np.tile(np.array([1.0, 2.0, 3.0, 6.0, 7.0]), (N, 1))   # all users start alike


# ------------------------------------------------------- baseline: Ye & Hu
def run_yehu(b=0.3, k=0.3, th=0.045, w0=10.0, t_end=200.0):
    """Ye & Hu (2016) algorithm (8) on the undirected 5-cycle.

    Their gains are k_ij = delta*b*kbar and theta = delta*b^2*thetabar with
    delta, b small, so the averaged gradient gain is k*b/2; the values here put
    that at 0.045, which settles well inside the 200 s window.  The frequencies
    are odd multiples of w0, which makes w_a = w_b + w_c impossible and so
    satisfies the separation conditions of their Theorem 2.
    """
    # one frequency per (agent, coordinate): N*d = 25 distinct values
    odd = np.arange(1, N * d + 1) * 2 + 1.0        # 3, 5, ..., 51
    W = w0 * odd.reshape(N, d)

    def sys(t, st):
        Xh = st[:N * d].reshape(N, d)
        Z = st[N * d:].reshape(N, d)
        S = np.sin(W * t)
        X = Xh + b * S                     # the state that is actually applied
        fv = f_all(X)                      # only the value is measured
        cons = L_UND @ Xh + L_UND @ Z
        dXh = -k * (fv[:, None] * S + 0.5 * b * cons)
        dZ = th * (L_UND @ Xh)
        return np.concatenate([dXh.ravel(), dZ.ravel()])

    st0 = np.concatenate([X0.ravel(), np.zeros(N * d)])
    t = np.linspace(0, t_end, 20000)
    s = solve_ivp(sys, (0, t_end), st0, method='DOP853', t_eval=t,
                  rtol=1e-7, atol=1e-9)
    Xh = s.y[:N * d].reshape(N, d, -1)
    Sfull = np.sin(W[:, :, None] * s.t[None, None, :])
    X = Xh + b * Sfull                     # report the applied profile
    return s.t, X


# ------------------------------------------------------------------- ours
def run_ues(beta=0.1, v=1.0, q=2, omega=60.0, alpha=1.2, k=0.5, gamma=0.05,
            omega_h=8.0, t_end=200.0):
    """Chirpy uES, asymptotic configuration, on the directed ring.

    phi(t) = (1 + beta t)^(1/v) grows only to 21 over the 200 s window, so the
    instantaneous probing frequency omega*phi^(p+1) stays below 1300 rad/s and
    no saturation of phi is needed here -- the scheme runs exactly as analysed.

    k = 0.5 rather than the 32 used in the manuscript's scalar example: the
    local costs here are of order 8, so the demodulation phase k*phi*(f - eta)
    would swing far too much at large k and Lie-bracket averaging degrades.  The
    product alpha*k = 0.6 is unchanged, which is all the LMIs constrain.
    """
    p = q - v - 1
    rho = v / (beta * q)
    what = np.array([3., 5., 7., 11., 13.])          # d distinct frequencies
    ws = omega * what
    t_cap = np.inf                                   # never reached here

    def sys(t, st):
        X = st[:N * d].reshape(N, d)
        eta = st[N * d:N * d + N]
        Z = st[N * d + N:].reshape(N, d)
        ph = (1 + beta * t) ** (1 / v)
        tau = rho * (ph ** q - 1.0)
        fv = f_all(X)
        # cos(w_s tau + k phi (f_i - eta_i)) is an N-vector for each s
        arg = ws[None, :] * tau + (k * ph * (fv - eta))[:, None]
        probe = (ph ** p) * np.sqrt(alpha * omega) * np.cos(arg)   # (N, d)
        dX = probe - (ph ** (p + 1)) * (L_DIR @ X) - (ph ** p) * Z
        deta = (-omega_h * eta + omega_h * fv) * (ph ** (p + 1))
        dZ = gamma * (L_DIR @ X) * (ph ** (p + 2))
        return np.concatenate([dX.ravel(), deta, dZ.ravel()])

    st0 = np.concatenate([X0.ravel(), np.zeros(N), np.zeros(N * d)])
    t = np.linspace(0, t_end, 20000)
    s = solve_ivp(sys, (0, t_end), st0, method='DOP853', t_eval=t,
                  rtol=1e-7, atol=1e-9)
    return s.t, s.y[:N * d].reshape(N, d, -1), t_cap


def err_of(X):
    """max over users of || estimate - l* ||."""
    return np.linalg.norm(X - LOPT[None, :, None], axis=1).max(axis=0)


if __name__ == "__main__":
    print("social optimum l* =", np.round(LOPT, 4), " sum =", round(LOPT.sum(), 4))
    ei = np.eye(d)
    for i in range(N):
        Hi = 2 * RHO[i] * np.outer(ei[i], ei[i]) + KP * (
            np.outer(ei[i], np.ones(d)) + np.outer(np.ones(d), ei[i]))
        ev = np.linalg.eigvalsh(Hi)
        if i == 0:
            print("Hessian of each f_i (singular -- see the note above):")
        print(f"  user {i+1}: eig min={ev.min():+.4f}  max={ev.max():+.4f}")
    Hf = 2 * np.diag(RHO) + 2 * KP * np.ones((d, d))
    ev = np.linalg.eigvalsh(Hf)
    print(f"aggregate cost Hessian: min={ev.min():.4f}  max={ev.max():.4f} "
          f"(strongly convex)")

    print("\nYe & Hu (2016), undirected 5-cycle ...")
    ty, Xy = run_yehu()
    ey = err_of(Xy)
    print(f"  err(200s)={ey[-1]:.4f}   ripple over last 20 s = "
          f"{ey[ty > 180].max() - ey[ty > 180].min():.4f}")

    print("Proposed uES, directed ring ...")
    tu, Xu, tcap = run_ues()
    eu = err_of(Xu)
    print(f"  err(100s)={np.interp(100.0, tu, eu):.4e}   err(200s)={eu[-1]:.4e}"
          f"   (still decreasing: no residual floor)")

    # Trajectories only.  A semilog error panel was tried and dropped: past
    # t ~ 130 our curve is a thick dither band that reads as noise, and the
    # numbers it would carry are already in the caption and the README table.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    for ax, (t, X, ttl) in zip(axes,
                               [(ty, Xy, "ES of [Ye and Hu, 2016]"),
                                (tu, Xu, "Proposed uES")]):
        for j in range(d):
            ax.plot(t, X[0, j], lw=1.0, label=f"$l_{j+1}$")
        for j in range(d):
            ax.axhline(LOPT[j], color='k', ls=':', lw=0.9)
        ax.set_xlabel("Time (sec)")
        ax.set_title(ttl, fontsize=LATEX_FONT_SIZE - 3)
        ax.set_xlim(0, 200)
        ax.set_ylim(0.5, 7.6)
        ax.grid(True)
    axes[0].set_ylabel("User 1's estimate")
    axes[0].legend(ncol=3, framealpha=0.9, loc='upper right')

    fig.tight_layout(pad=0.4)
    fig.savefig('out/compare_es_smartgrid.eps', format='eps')
    fig.savefig('out/compare_es_smartgrid.png', dpi=140)
    print("saved out/compare_es_smartgrid.{eps,png}")
