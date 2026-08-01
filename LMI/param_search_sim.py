"""
Convergence of the d = 3 example for several LMI-feasible parameter sets.

check_lmi.py shows that the parameters currently used in Section VI-A
(alpha = k = gamma = 1, v = 2, beta = 1) do NOT satisfy the LMIs of
Proposition 1. This script simulates the candidates that do satisfy them, so
that a replacement can be chosen on the basis of how clearly the figure still
shows unbiased convergence.

Key point behind the choice of candidates: the LMIs only constrain the ratio
beta/v, whereas the growth of xi(t) = (1 + beta t)^{1/v} is governed mainly by
the exponent 1/v. Keeping v at its smallest admissible value (v = 2) and
lowering beta therefore buys feasibility at the least cost in growth:
beta/v = 0.25 with v = 2 gives xi(100) = 7.1, while the same ratio with v = 10
gives only 1.5.

Run:  python3 param_search_sim.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.ndimage import maximum_filter1d

SIM = "../fig2_beta_3d.py"
base = open(SIM).read().split("def make_system(")[0]
ns = {}
exec(base, ns)
N, d, L, x_star = ns["N"], ns["d"], ns["L"], ns["x_star"]
omega_h, omega_s, local_cost = ns["omega_h"], ns["omega_s"], ns["local_cost"]

X0 = np.array([[-1.0, 0.0, 2.0], [0.0, 3.0, -1.0], [1.0, -1.0, 4.0],
               [4.0, 5.0, 0.0], [5.0, 1.0, 5.0]])


def run(beta, v, gamma, alpha, k, T):
    def xi(t):
        return (1.0 + beta * t) ** (1.0 / v)

    def sys(t, st):
        X = st[:N * d].reshape(N, d)
        eta = st[N * d:N * d + N]
        Z = st[N * d + N:].reshape(N, d)
        xt = xi(t)
        f = np.array([local_cost(X[i], i) for i in range(N)])
        dX = np.empty((N, d))
        for s in range(d):
            dX[:, s] = (np.sqrt(alpha * omega_s[s]) / xt) * \
                       np.cos(omega_s[s] * t + k * xt * (f - eta))
        LX = L @ X
        dX += -LX - Z / xt
        return np.concatenate([dX.ravel(),
                               -omega_h * eta + omega_h * f,
                               (gamma * LX * xt).ravel()])

    y0 = np.concatenate([X0.ravel(), np.zeros(N), np.zeros(N * d)])
    sol = solve_ivp(sys, (0, T), y0, method="DOP853", dense_output=True,
                    rtol=1e-8, atol=1e-10)
    t = np.linspace(0, T, 12000)
    X = sol.sol(t)[:N * d].reshape(N, d, -1)
    err = np.linalg.norm(X - x_star[None, :, None], axis=(0, 1))
    win = max(1, int(len(t) * (2 * np.pi / omega_s[0]) / T))
    return t, maximum_filter1d(err, size=win)


if __name__ == "__main__":
    T = 200.0
    # (beta, v, gamma, alpha, k, label, LMI margin from check_lmi.py)
    cases = [
        (0.0, 2.0, 0.02, 0.6, 1.0, "bounded ES  $\\beta=0$", None),
        (1.0, 2.0, 1.00, 1.0, 1.0, "current figure  $\\beta=1,\\gamma=1,\\alpha k=1$", -0.201),
        (0.5, 2.0, 0.02, 0.6, 1.0, "A  $\\beta=0.5,\\gamma=0.02,\\alpha=0.6,k=1$", +0.0040),
        (0.4, 2.0, 0.10, 0.5, 1.0, "B  $\\beta=0.4,\\gamma=0.1,\\alpha=0.5,k=1$", +0.0103),
        (0.4, 2.0, 0.10, 1.0, 0.5, "C  $\\beta=0.4,\\gamma=0.1,\\alpha=1,k=0.5$", +0.0103),
    ]

    plt.rcParams.update({"font.size": 12, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(8, 5))
    for beta, v, gamma, alpha, k, lab, marg in cases:
        t, e = run(beta, v, gamma, alpha, k, T)
        style = dict(lw=1.2)
        if marg is None:
            style.update(color="0.4", ls="--")
        elif marg < 0:
            style.update(color="tab:orange")
        ax.semilogy(t, e, label=lab, **style)
        tag = "infeasible" if (marg is not None and marg < 0) else \
              ("feasible" if marg is not None else "baseline")
        print(f"{lab:52s} xi(T)={((1+beta*T)**(1/v)):6.2f}  "
              f"err(T)={e[-1]:.3e}  [{tag}]", flush=True)

    ax.set_xlabel("Time (sec)")
    ax.set_ylabel(r"$\|\mathbf{x}(t)-\mathbf{1}_N\otimes x^*\|$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout(pad=0.3)
    fig.savefig("param_search_sim.png", dpi=150)
    print("\nsaved param_search_sim.png")
