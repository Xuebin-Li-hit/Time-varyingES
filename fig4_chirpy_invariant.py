import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

LATEX_FONT_SIZE = 22  # 对应论文正文字体大小（单位：pt）
legendfontsize = 15

# ================ Matplotlib 全局设置 ================
plt.rcParams.update({
    'font.size': LATEX_FONT_SIZE,       # 全局字体大小
    'axes.titlesize': LATEX_FONT_SIZE,  # 子图标题
    'axes.labelsize': LATEX_FONT_SIZE,  # 坐标轴标签
    'xtick.labelsize': LATEX_FONT_SIZE - 1,  # X刻度（稍小1pt）
    'ytick.labelsize': LATEX_FONT_SIZE - 1,  # Y刻度
    'legend.fontsize': legendfontsize - 1,  # 图例
    'font.family': 'serif',             # 匹配论文字体
    'font.serif': ['Times New Roman'],  # 推荐期刊字体
})

os.makedirs('out', exist_ok=True)

# ---------------------------------------------------------------------------
# Chirp phase.  The probing argument in (14) is omega_s * tau with
#     tau = t_0 + rho (phi^q(t) - 1),
# so the phase is omega * rho * (phi^q - 1).  An earlier version of this
# script multiplied that by t as well, which is dimensionally wrong (rad*s)
# and destroys the chirp demodulation.
#
# Growth rates are unchanged from the submitted version: all three
# configurations have rho = 2.5, hence 1/(q rho) = 0.2, so they share one
# LMI certificate (gamma = 0.05, alpha*k = 0.6, m = 1.75, M = 4: margin
# +0.013).  What changed are the probing gains.  The LMIs of Proposition 1
# constrain only the product alpha*k, while the residual dither in the state is
#     ripple(t) = sqrt(alpha/omega) / phi(t),
# so trading alpha down against k up at fixed alpha*k shrinks the ripple at no
# cost in the certificate: alpha = 0.6, k = 1, omega = 10 gives 0.245, whereas
# alpha = 0.01875, k = 32, omega = 40 gives 0.0217.  Raising k further is
# counter-productive -- at k = 128 the phase k*phi*(f - eta) wraps often enough
# that Lie-bracket averaging degrades at finite omega.
# ---------------------------------------------------------------------------

# 通用参数
N = 5        # 智能体数量
n = 1        # 每个智能体的状态维度
omega_h = 8
omega = 40         # 扰动频率
alpha = 0.6 / 32   # alpha_i
k = 32             # alpha * k = 0.6
a = 1        # proportional consensus gain
b = 0.05     # integral consensus gain (gamma)
q = 2

# 通信拓扑（有向环图，权重平衡且强连通）
A = np.array(
    [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
    ]
)  # 邻接矩阵
D = np.diag(np.sum(A, axis=1))
L = D - A  # 拉普拉斯矩阵

# 目标函数：各智能体的本地目标函数（c_i为本地最优值）
c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 各智能体的本地最优值

# f_i(x_i) = (x_i - c_i)^2 + ln(1 + (x_i - c_i)^2).
# The earlier variant used sin((x_i - c_i)^2), whose second derivative dips to
# -130 over the operating range, so each f_i violated the m_i-strong-convexity
# and Lipschitz-gradient requirements of Assumption 1. The ln term keeps
# min f_i'' = 1.75 > 0 and matches the cost family used in Subsection VI-A.
def f_vec(x):
    return (x - c) ** 2 + np.log1p((x - c) ** 2)


# 初始条件
hat_X0 = np.array([-1.0, 0.0, 1.0, 4.0, 5.0])
initial_state = np.concatenate([hat_X0, np.zeros(N * n), np.zeros(N * n)])


def simulate(phi_raw, p, rho, t_end, phi_cap):
    """Integrate (14).

    phi is saturated at phi_cap.  Past the cap the warped time tau must keep
    advancing at the frozen rate d(tau)/dt = phi_cap^(p+1); freezing tau
    instead would stop the probing oscillation altogether.  The cap binds only
    for the prescribed-time configuration, where phi -> infinity as
    t -> t_0 + T (see Remark 8); for the asymptotic and exponential ones it is
    placed beyond the plotted horizon and never activates.
    """
    t_cap = np.inf
    for tt in np.linspace(1e-9, t_end, 200000):
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
            omega * tau + k * ph * (f_vec(x) - eta)
        ) + (-a * (ph ** (p + 1)) * L_hatX - z * (ph ** p))
        d_eta = (-omega_h * eta + omega_h * f_vec(x)) * (ph ** (p + 1))
        d_z = b * L_hatX * (ph ** (p + 2))
        return np.concatenate([d_hatx, d_eta, d_z])

    t_eval = np.linspace(0, t_end, 10000)
    sol = solve_ivp(system, (0, t_end), initial_state, method="DOP853",
                    t_eval=t_eval, rtol=1e-7, atol=1e-9)
    x = sol.y[: N * n]
    err = np.abs(x - np.mean(c)).max(axis=0)
    print(f"  p={p:.1f} rho={rho:.2f} t_cap={t_cap:.2f} "
          f"final |x-x*|={err[-1]:.3e} nfev={sol.nfev}", flush=True)
    return sol.t, x


def draw(t, x, title, fname, ylabel=False, legend=False, xticks=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    for i in range(N):
        ax.plot(t, x[i], label=f"Agent {i + 1}")
    ax.axhline(np.mean(c), color="r", linestyle="--", label="Optimum")
    ax.set_xlabel("Time (sec)")
    if ylabel:
        ax.set_ylabel("State ${x}_i(t)$")
    ax.set_title(title)
    if legend:
        ax.legend(ncol=2, framealpha=0.9, loc="lower right")
    if xticks is not None:
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks(xticks)
    ax.grid(True)
    plt.tight_layout(pad=0.2)
    plt.savefig(f'out/{fname}.eps', format='eps')
    plt.close(fig)


# ================ 第一部分：Asymptotic uES ================
# phi(t) = (1 + beta t)^(1/v),  p = q - v - 1,  rho = v/(beta q)
print("Computing Asymptotic uES...")
belta, v = 0.1, 0.5
# phi(30) = 16.0, so a cap at 16.1 never activates within the horizon.
t1, x1 = simulate(lambda t: (1 + belta * t) ** (1 / v), q - v - 1,
                  v / (belta * q), t_end=30.0, phi_cap=16.1)
draw(t1, x1, "Asymptotic uES", "chirpy_asy_invariant", ylabel=True, legend=True)

# ================ 第二部分：Exponential uES ================
# phi(t) = e^(lambda t),  p = q - 1,  rho = 1/(lambda q)
print("Computing Exponential uES...")
lambda_ = 0.2
# phi(12) = e^2.4 = 11.0, so a cap at 11.1 never activates within the horizon.
t2, x2 = simulate(lambda t: np.exp(lambda_ * t), q - 1, 1 / (lambda_ * q),
                  t_end=12.0, phi_cap=11.1)
draw(t2, x2, "Exponential uES", "chirpy_exp_invariant")

# ================ 第三部分：Prescribed-time uES ================
# phi(t) = (T/(T-t))^(1/varrho),  p = q + varrho - 1,  rho = varrho T/q
print("Computing Prescribed-time uES...")
T, varrho = 5.0, 1.0
# phi blows up at t = T, so the cap is unavoidable here (Remark 8):
# phi_cap = 10 is reached at t = 4.5 s.
t3, x3 = simulate(lambda t: (T / max(T - t, 1e-12)) ** (1 / varrho),
                  q + varrho - 1, varrho * T / q,
                  t_end=5.0, phi_cap=10.0)
draw(t3, x3, "Prescribed-time uES", "chirpy_pt_invariant",
     xticks=[0, 1, 2, 3, 4, 5])   # make the prescribed time T = 5 visible

print("All simulations completed!")
