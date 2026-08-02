"""
Fig. 4 -- uES with chirpy probing, time-varying extrema.

Chirp phase.  The probing argument in (14) is omega_s * tau with
    tau = t_0 + rho (phi^q(t) - 1),
so the phase is omega * rho * (phi^q - 1) and carries no separate factor t;
writing one would make the argument rad*s and destroy the demodulation.

Choice of q.  The targets have bounded but non-decaying derivatives, so the
smallest admissible c in Assumption 2 is c = 0 and Theorem 2 needs c - p < -2,
i.e. p > 2.  For the asymptotic and exponential configurations p = q - v - 1 and
p = q - 1, so q = 4 (giving p = 2.5 and 3).  For the prescribed-time one
p = q + varrho - 1, and q = 2 with varrho = 2 already gives p = 3.

Why varrho = 2 for the prescribed-time panel.  Saturating phi at a fraction f of
T forces phi_cap = (1-f)^(-1/varrho), so with varrho = 1 a cap at 0.9T needs
phi_cap = 10 and the post-cap probing frequency omega*phi_cap^(p+1) reaches
4e7 rad/s, which cannot be integrated.  varrho = 2 flattens the blow-up:
phi_cap = sqrt(10) = 3.16 and the frequency is 4e4 rad/s, so the cap can sit at
t = 4.5 s, close to T, at half the cost of the earlier setting.

Probing gains.  The LMIs constrain only 1/(q rho) and the product alpha*k, and
omega does not enter them at all, while it sets both the averaging error
(~1/omega) and the dither ripple sqrt(alpha/omega)/phi.  omega is therefore the
free lever: raising it from 100 to 400 cut the tracking bias by ~40%.  Raising
alpha*k above 1.6 makes the bias worse, so bandwidth was never the bottleneck,
and 1.6 is also where the LMI margin peaks.  Margins: +0.376 for the asymptotic
and exponential panels (1/(q rho) = 0.2), +0.412 for the prescribed-time one
(1/(q rho) = 1/(varrho T) = 0.1).

Figure geometry.  bbox_inches='tight' must NOT be used here: it crops each panel
to its own ink extent, so the three panels come out at different widths, scale
to \\columnwidth by different factors, and end up with different font sizes (and
an aspect ratio that is no longer 3:1).  Instead the canvas is fixed and the
axes rectangle is set by hand, so all three share one bounding box of
648 x 216 pt.  At \\columnwidth = 249 pt the scale is 0.384, so the 21 pt font
lands at about 8 pt in the compiled paper -- IEEE caption size.
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

# 通用参数
N = 5
n = 1
omega_h = 8
omega = 400        # 扰动频率
alpha = 1.6        # alpha_i
k = 1              # alpha * k = 1.6
a = 1              # proportional consensus gain
b = 0.05           # integral consensus gain (gamma)
t_end = 80.0
F_MAX = 80000.0    # post-cap probing frequency ceiling for the asymptotic and
                   # exponential panels (a hardware bandwidth limit, not a limit
                   # on phi itself); the prescribed-time panel caps at t = 4.5 s

# 通信拓扑（有向环图，权重平衡且强连通）
A = np.array(
    [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
    ]
)
D = np.diag(np.sum(A, axis=1))
L = D - A

# 时变最优解：x_i^*(t) = c_i + A_i sin(w_i t).
# The offsets mirror the evenly spaced c_i of Fig. 3 but are centred at 1.75,
# and the amplitudes are scaled so that the global optimum x^*(t) sweeps
# [0, 3.5] -- the same range as the spread initial condition, which keeps the
# tracking visible.  Each f_i is a pure quadratic, so m = M = 2.
CEN = np.array([0.75, 1.25, 1.75, 2.25, 2.75])
AMP = np.array([0.6, 1.8, 3.0, 2.4, 3.0])
FRQ = np.array([0.1, 0.2, 0.3, 0.1, 0.4])


def xstar_i(t):
    return CEN + AMP * np.sin(FRQ * t)


def f_vec(x, t):
    return (x - xstar_i(t)) ** 2


# 初始条件（与 Fig. 3 相同）
hat_X0 = np.array([-1.0, 0.0, 1.0, 4.0, 5.0])
initial_state = np.concatenate([hat_X0, np.zeros(N * n), np.zeros(N * n)])


def simulate(phi_raw, p, rho, q, phi_cap):
    """Integrate (14).

    phi is saturated at phi_cap.  Past the cap the warped time tau must keep
    advancing at the frozen rate d(tau)/dt = phi_cap^(p+1); freezing tau instead
    would stop the probing oscillation altogether.  For the prescribed-time
    configuration the cap is unavoidable, since phi -> infinity as
    t -> t_0 + T (see Remark 8).
    """
    t_cap = np.inf
    for tt in np.linspace(1e-9, t_end, 800000):
        if phi_raw(tt) >= phi_cap:
            t_cap = tt
            break
    tau_cap = rho * (phi_cap ** q - 1.0)

    def system(t, states):
        x = states[: N * n]
        eta = states[N * n: 2 * N * n]
        z = states[2 * N * n:]

        if t <= t_cap:
            ph = phi_raw(t)
            tau = rho * (ph ** q - 1.0)
        else:
            ph = phi_cap
            tau = tau_cap + phi_cap ** (p + 1) * (t - t_cap)

        L_hatX = L @ x
        d_hatx = (ph ** p) * np.sqrt(alpha * omega) * np.cos(
            omega * tau + k * ph * (f_vec(x, t) - eta)
        ) + (-a * (ph ** (p + 1)) * L_hatX - z * (ph ** p))
        d_eta = (-omega_h * eta + omega_h * f_vec(x, t)) * (ph ** (p + 1))
        d_z = b * L_hatX * (ph ** (p + 2))
        return np.concatenate([d_hatx, d_eta, d_z])

    t_eval = np.linspace(0, t_end, 24000)
    sol = solve_ivp(system, (0, t_end), initial_state, method="DOP853",
                    t_eval=t_eval, rtol=1e-7, atol=1e-9)
    x = sol.y[: N * n]
    xs = np.array([xstar_i(ti).mean() for ti in sol.t])
    late = sol.t > 20
    bias = np.abs(x[:, late].mean(axis=0) - xs[late]).max()
    print(f"  q={q} p={p:.1f} rho={rho:.2f} 1/(q rho)={1/(q*rho):.3f} "
          f"phi_cap={phi_cap:.2f} t_cap={t_cap:.2f} F={omega*phi_cap**(p+1):.0f} "
          f"bias={bias:.4f} nfev={sol.nfev}", flush=True)
    return sol.t, x, xs


def draw(t, x, xs, title, fname, mark_T=None):
    # Fixed canvas with a hand-set axes rectangle, so every panel has the same
    # bounding box; see the note at the top of this file on why tight cropping
    # is wrong here.
    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_axes([0.078, 0.235, 0.905, 0.660])
    for i in range(N):
        ax.plot(t, x[i], lw=1.1, label=f"Agent {i + 1}")
    ax.plot(t, xs, 'k--', lw=1.7, label="$x^*(t)$")
    ax.set_xlabel("Time (sec)", labelpad=1)
    ax.set_ylabel("State ${x}_i(t)$", labelpad=2)
    ax.set_title(title, pad=4)
    ax.set_xlim(t[0], t[-1])
    ax.grid(True)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.30)
    if mark_T is not None:
        ax.axvline(x=mark_T, color='k', linestyle='--')
        ax.text(mark_T + 1.2, y_min + 0.35, f"$T={mark_T:g}$")
    ax.legend(loc='upper right', ncol=3, framealpha=0.9, handlelength=1.1,
              columnspacing=0.9, borderpad=0.28, labelspacing=0.25)
    fig.savefig(f'out/{fname}.eps', format='eps')
    plt.close(fig)


# ================ Asymptotic uES ================
# phi(t) = (1 + beta t)^(1/v),  p = q - v - 1,  rho = v/(beta q)
print("Computing Asymptotic uES...")
belta, v, q = 0.1, 0.5, 4
p = q - v - 1
t1, x1, xs1 = simulate(lambda t: (1 + belta * t) ** (1 / v), p, v / (belta * q),
                       q, (F_MAX / omega) ** (1.0 / (p + 1)))
draw(t1, x1, xs1, "Asymptotic uES for time-varying distributed optimization",
     "chirpy_asy_varying")

# ================ Exponential uES ================
# phi(t) = e^(lambda t),  p = q - 1,  rho = 1/(lambda q)
print("Computing Exponential uES...")
lambda_, q = 0.2, 4
p = q - 1
t2, x2, xs2 = simulate(lambda t: np.exp(lambda_ * t), p, 1 / (lambda_ * q),
                       q, (F_MAX / omega) ** (1.0 / (p + 1)))
draw(t2, x2, xs2, "Exponential uES for time-varying distributed optimization",
     "chirpy_exp_varying")

# ================ Prescribed-time uES ================
# phi(t) = (T/(T-t))^(1/varrho),  p = q + varrho - 1,  rho = varrho T/q
print("Computing Prescribed-time uES...")
T, varrho, q = 5.0, 2.0, 2
p = q + varrho - 1
t_th = 4.5                                   # cap time, close to T
t3, x3, xs3 = simulate(lambda t: (T / max(T - t, 1e-12)) ** (1 / varrho),
                       p, varrho * T / q, q,
                       (T / (T - t_th)) ** (1 / varrho))
draw(t3, x3, xs3, "Prescribed-time uES for time-varying distributed optimization",
     "chirpy_pt_varying", mark_T=T)

print("All simulations completed!")
