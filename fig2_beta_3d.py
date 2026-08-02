"""
Three-dimensional (d = 3) version of the constant-frequency example.

Produces the two panels of Fig. 2:
    out/beta3d=0.eps   beta = 0   -> bounded (biased) ES
    out/beta3d=1.eps   beta = 0.2 -> asymptotic uES

Both panels show the tracking error ||x(t) - 1_N (x) x*|| on a semilogarithmic
axis, smoothed by a sliding maximum over one probing period, and share the
same y-range. Since the quantity plotted is a norm, the figure looks the same
for any state dimension d, so this drop-in replaces the scalar example while
answering the request for a higher-dimensional (d >= 3) experiment.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.ndimage import maximum_filter1d

# ---------------------------------------------------------------- style
LATEX_FONT_SIZE = 22
legendfontsize = 15
plt.rcParams.update({
    'font.size': LATEX_FONT_SIZE,
    'axes.titlesize': LATEX_FONT_SIZE,
    'axes.labelsize': LATEX_FONT_SIZE,
    'xtick.labelsize': LATEX_FONT_SIZE - 1,
    'ytick.labelsize': LATEX_FONT_SIZE - 1,
    'legend.fontsize': legendfontsize - 1,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
})

# ---------------------------------------------------------------- setup
# These values satisfy the three LMIs of Proposition 1 with margin +0.0126;
# see LMI/check_lmi.py.  The certificate is tight in both beta/v and alpha*k
# for this cost family (m = 1.75, M = 4), which is what fixes them at 0.1 and
# 0.4: raising either much above that loses feasibility.
#
# The horizon follows from beta/v.  With beta = 0.2, v = 2 the scaling reaches
# xi(400) = 9, so 400 s is what it takes for the unbiasing to show clearly
# against the constant-amplitude baseline.
N = 5            # agents
d = 3            # state dimension of each agent
k = 1.0
omega_h = 10.0
omega = 10.0     # base probing frequency
alpha = 0.4      # alpha * k = 0.4
v = 2.0
a = 1.0          # proportional consensus gain
b = 0.1          # integral consensus gain  (gamma in the paper)
t_span = (0.0, 400.0)

# Distinct integer multipliers, one per coordinate (Theorem 1 requires
# hat_omega_s in N, pairwise distinct).  Beyond that condition, the choice
# matters in the multivariable case: the demodulation sits inside the phase,
# cos(omega_s t + k xi (h - eta)), so the probe is a phase-modulated carrier
# whose sidebands sit at omega_s + sum_l n_l omega_l.  Any low-order integer
# relation among the hat_omega_s puts a sideband at DC, i.e. a persistent
# drift in that coordinate.  With [1, 2, 5] (where 2 = 1 + 1) the optimality
# gap of the second coordinate is 0.168; with [3, 5, 7], which admits no such
# low-order relation, it drops to 0.013.
hat_w = np.array([3.0, 5.0, 7.0])
omega_s = omega * hat_w

# directed ring: weight-balanced and strongly connected (Assumption 3)
A = np.array([[0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0]], dtype=float)
L = np.diag(A.sum(axis=1)) - A

# local minimisers c_i in R^3, spread out so the agents genuinely disagree
c = np.array([[1.0, 2.0, 1.0],
              [2.0, 1.0, 3.0],
              [3.0, 3.0, 2.0],
              [4.0, 2.0, 4.0],
              [5.0, 4.0, 3.0]])


def local_cost(xi_vec, i):
    """f_i(x) = ||x - c_i||^2 + ln(1 + ||x - c_i||^2)   (C^3, strongly convex)"""
    r2 = np.sum((xi_vec - c[i]) ** 2)
    return r2 + np.log1p(r2)


def global_cost(x_vec):
    return sum(local_cost(x_vec, i) for i in range(N))


# the ln terms move the optimum away from the centroid, so solve for it
x_star = minimize(global_cost, c.mean(axis=0), method='BFGS',
                  options={'gtol': 1e-12}).x
print(f"x* = {np.array2string(x_star, precision=6)}")


def make_system(belta):
    def xi(t):
        return (1.0 + belta * t) ** (1.0 / v)

    def system(t, states):
        X = states[:N * d].reshape(N, d)          # agent states
        eta = states[N * d: N * d + N]            # one filter state per agent
        Z = states[N * d + N:].reshape(N, d)      # integral states

        xi_t = xi(t)
        f = np.array([local_cost(X[i], i) for i in range(N)])   # h(x, zeta)

        # extremum-seeking probing: coordinate s is excited at frequency omega_s
        dX = np.empty((N, d))
        for s in range(d):
            dX[:, s] = (np.sqrt(alpha * omega_s[s]) / xi_t) \
                       * np.cos(omega_s[s] * t + k * xi_t * (f - eta))

        LX = L @ X                                 # (L (x) I_d) x
        dX += -a * LX - Z / xi_t                   # PI consensus
        d_eta = -omega_h * eta + omega_h * f
        dZ = b * LX * xi_t
        return np.concatenate([dX.ravel(), d_eta, dZ.ravel()])

    return system


def run(belta):
    X0 = np.array([[-1.0, 0.0, 2.0],
                   [0.0, 3.0, -1.0],
                   [1.0, -1.0, 4.0],
                   [4.0, 5.0, 0.0],
                   [5.0, 1.0, 5.0]])
    y0 = np.concatenate([X0.ravel(), np.zeros(N), np.zeros(N * d)])
    # the dynamics are oscillatory rather than stiff, so the explicit DOP853
    # is ~8x faster than Radau here and gives identical trajectories
    sol = solve_ivp(make_system(belta), t_span, y0, method='DOP853',
                    dense_output=True, rtol=1e-8, atol=1e-10)
    t = np.linspace(t_span[0], sol.t[-1], 28000)
    X = sol.sol(t)[:N * d].reshape(N, d, -1)
    err = np.linalg.norm(X - x_star[None, :, None], axis=(0, 1))
    return t, err


t0, e0 = run(0.0)   # bounded ES
t1, e1 = run(0.2)   # asymptotic uES,  beta/v = 0.1

# envelope over roughly one period of the slowest dither
ENV_WIN = max(1, int(len(t0) * (2 * np.pi / omega_s[0]) / (t0[-1] - t0[0])))
e0 = maximum_filter1d(e0, size=ENV_WIN)
e1 = maximum_filter1d(e1, size=ENV_WIN)

# Fixed canvas and a common axes rectangle for both panels: tight_layout would
# shrink the data area of whichever panel carries the y-label, so the two
# subfigures would end up with different plot widths once LaTeX scales them to
# equal column fractions.
RECT = [0.235, 0.175, 0.700, 0.735]   # right edge 0.935: room for the "400" tick
lo = min(e0.min(), e1.min())
hi = max(e0.max(), e1.max())

for _t, _e, _title, _ylab, _name in (
        (t0, e0, 'Bounded ES',    True,  'out/beta3d=0.eps'),
        (t1, e1, 'Asymptotic uES', False, 'out/beta3d=1.eps')):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_axes(RECT)
    ax.semilogy(_t, _e, color='C0')
    ax.set_xlabel('Time (sec)')
    if _ylab:
        ax.set_ylabel(r'$\|\mathbf{x}(t)-\mathbf{1}_N\otimes x^*\|$', fontsize=16)
    ax.set_title(_title)
    ax.set_xlim(t_span)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_ylim(lo / 2, hi * 2)
    ax.grid(True)
    fig.savefig(_name, format='eps')
    plt.close(fig)

print(f"final error: bounded ES {e0[-1]:.3e} | asymptotic uES {e1[-1]:.3e}")
