"""
Comparison against a classical *gradient-based* distributed method.

Baseline: S. S. Kia, J. Cortes, S. Martinez, "Distributed convex optimization
via continuous-time coordination algorithms with discrete-time communication",
Automatica 55 (2015) 254-264, algorithm (4):

    v_dot = alpha*beta*L*x
    x_dot = -alpha*grad_f(x) - beta*L*x - v ,      sum_i v_i(0) = 0,

which converges exponentially over strongly connected, weight-balanced
digraphs -- the same class our Assumption 3 asks for, so the two schemes can be
run on the identical topology with no change of graph.

Test problem: the example of Section VI-B of the manuscript,
    f_i(x) = (x - i)^2 + ln(1 + (x - i)^2),   i = 1..5,   x* = 3.

Two points the comparison is meant to make:

  1. The Kia et al. law evaluates grad f_i, so it needs the analytic expression
     of each local cost.  Our scheme only samples the value f_i(x_i) and never
     forms a gradient, which is the setting the manuscript addresses.

  2. The Kia et al. law converges exponentially at a rate set by (alpha, beta,
     m, M) -- it can be made faster by retuning, but the settling time cannot be
     *assigned*.  Our chirpy scheme reaches the optimum within a user-specified
     T, here T = 5 s, independently of the initial condition.

Outputs out/compare_gradient.eps (and .png for quick viewing).
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
    'legend.fontsize': LATEX_FONT_SIZE - 6,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
})
os.makedirs('out', exist_ok=True)

N = 5
A = np.array([[0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0]], float)      # directed ring: weight-balanced
L = np.diag(A.sum(1)) - A
C = np.array([1., 2., 3., 4., 5.])
XSTAR = 3.0
X0 = np.array([-1., 0., 1., 4., 5.])


def f_vec(x):
    d = x - C
    return d * d + np.log1p(d * d)


def grad_f(x):
    """Analytic gradient -- available to the baseline, NOT to our scheme."""
    d = x - C
    return 2 * d + 2 * d / (1 + d * d)


# ----------------------------------------------------------------- baseline
def run_kia(alpha=1.0, beta=1.0, t_end=30.0):
    """Kia et al. (2015), algorithm (4)."""
    def sys(t, st):
        x, v = st[:N], st[N:]
        return np.concatenate([-alpha * grad_f(x) - beta * (L @ x) - v,
                               alpha * beta * (L @ x)])
    st0 = np.concatenate([X0, np.zeros(N)])          # sum_i v_i(0) = 0
    t = np.linspace(0, t_end, 6000)
    s = solve_ivp(sys, (0, t_end), st0, method='DOP853', t_eval=t,
                  rtol=1e-9, atol=1e-11)
    return s.t, s.y[:N]


# ------------------------------------------------------------------- ours
def run_ues_prescribed(T=5.0, varrho=1.0, q=2, omega=40.0, alpha=0.6 / 32,
                       k=32.0, gamma=0.05, omega_h=8.0, t_end=30.0,
                       phi_cap=10.0):
    """Chirpy uES, prescribed-time configuration (manuscript, Algorithm (14))."""
    p = q + varrho - 1
    rho = varrho * T / q
    phi_raw = lambda t: (T / max(T - t, 1e-12)) ** (1.0 / varrho)
    t_cap = np.inf
    for tt in np.linspace(1e-9, t_end, 400000):
        if phi_raw(tt) >= phi_cap:
            t_cap = tt
            break
    tau_cap = rho * (phi_cap ** q - 1.0)

    def sys(t, st):
        x, eta, z = st[:N], st[N:2 * N], st[2 * N:]
        if t <= t_cap:
            ph = phi_raw(t)
            tau = rho * (ph ** q - 1.0)
        else:
            ph = phi_cap
            tau = tau_cap + phi_cap ** (p + 1) * (t - t_cap)
        Lx = L @ x
        # only f_i(x_i) is measured here -- no gradient is ever formed
        probe = (ph ** p) * np.sqrt(alpha * omega) * np.cos(
            omega * tau + k * ph * (f_vec(x) - eta))
        return np.concatenate([probe - (ph ** (p + 1)) * Lx - (ph ** p) * z,
                               (-omega_h * eta + omega_h * f_vec(x)) * (ph ** (p + 1)),
                               gamma * Lx * (ph ** (p + 2))])

    st0 = np.concatenate([X0, np.zeros(2 * N)])
    t = np.linspace(0, t_end, 20000)
    s = solve_ivp(sys, (0, t_end), st0, method='DOP853', t_eval=t,
                  rtol=1e-7, atol=1e-9)
    return s.t, s.y[:N], t_cap


if __name__ == "__main__":
    T_PRESC = 5.0
    print("Kia et al. (2015), gradient-based ...")
    tk, xk = run_kia()
    ek = np.abs(xk - XSTAR).max(axis=0)
    print(f"  err(5s)={np.interp(5.0, tk, ek):.3e}  err(30s)={ek[-1]:.3e}")

    print("Proposed uES, prescribed-time T=5 ...")
    tu, xu, tcap = run_ues_prescribed(T=T_PRESC)
    eu = np.abs(xu - XSTAR).max(axis=0)
    print(f"  phi capped at t={tcap:.2f}s  err(5s)={np.interp(5.0, tu, eu):.3e} "
          f"err(30s)={eu[-1]:.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    ax = axes[0]
    for i in range(N):
        ax.plot(tk, xk[i], lw=1.1, label=f"Agent {i+1}")
    ax.axhline(XSTAR, color='r', ls='--', lw=1.4, label="Optimum")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("State ${x}_i(t)$")
    ax.set_title("Gradient-based [Kia et al., 2015]", fontsize=LATEX_FONT_SIZE - 3)
    ax.set_xlim(0, 30)
    ax.grid(True)
    ax.legend(ncol=2, framealpha=0.9, loc='lower right')

    ax = axes[1]
    for i in range(N):
        ax.plot(tu, xu[i], lw=1.1, label=f"Agent {i+1}")
    ax.axhline(XSTAR, color='r', ls='--', lw=1.4, label="Optimum")
    ax.axvline(T_PRESC, color='k', ls='--', lw=1.2)
    y0, y1 = ax.get_ylim()
    ax.text(T_PRESC + 0.8, y0 + 0.35, f"$T={T_PRESC:g}$")
    ax.set_xlabel("Time (sec)")
    ax.set_title("Proposed uES, prescribed time", fontsize=LATEX_FONT_SIZE - 3)
    ax.set_xlim(0, 30)
    ax.grid(True)

    fig.tight_layout(pad=0.4)
    fig.savefig('out/compare_gradient.eps', format='eps')
    fig.savefig('out/compare_gradient.png', dpi=140)
    print("saved out/compare_gradient.{eps,png}")
