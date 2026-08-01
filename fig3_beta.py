'''
FileDesc: 
Author: xuebin
Date: 2025-04-04 23:03:44
LastEditTime: 2025-05-27 23:10:28
Version: 
Usage: 
		- template
		-	js
		- props
		- event
		- method
        五个智能体的例子
'''


import numpy as np
import matplotlib
from scipy.ndimage import maximum_filter1d
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
N = 5  # 智能体数量
n = 1  # 每个智能体的状态维度
k = 1
lambda_ = 1
omega_h = 10
omega = 10  # 扰动频率（需满足频率互质）
t_span = (0, 100)  # 仿真时间范围
alpha = 1.0  # α_i
belta = 0
v = 2
a = 1
b = 1


# 通信拓扑（环形图，有向连通）
A = np.array([[0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0]])  # 邻接矩阵
D = np.diag(np.sum(A, axis=1))
L = D - A  # 拉普拉斯矩阵

# 目标函数：各智能体的本地目标函数（c_i为本地最优值）
c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 各智能体的本地最优值

def xi(t):
    return (1 + belta * t) ** (1 / v)


def local_cost(x_i, i):
    return (x_i - c[i]) ** 2 + np.sin((x_i - c[i]) ** 2)  # f_i(x_i) = (x_i - c_i)^2 + sin((x_i - c_i)^2)


def d_local_cost(x_i, i):
    diff = x_i - c[i]
    return 2 * diff + 2 * diff * np.cos(diff ** 2)  # d_f_i(x_i) = 2*(x_i - c_i) + 2*(x_i - c_i)*cos((x_i - c_i)^2)


def f_vec(x):
    return np.array([local_cost(x[0], 0), local_cost(x[1], 1), local_cost(x[2], 2), local_cost(x[3], 3), local_cost(x[4], 4)])


def globle_cost(x):
    return local_cost(x, 0) + local_cost(x, 1) + local_cost(x, 2) + local_cost(x, 3) + local_cost(x, 4)


def d_f_vec(x):
    return [d_local_cost(x[0], 0), d_local_cost(x[1], 1), d_local_cost(x[2], 2), d_local_cost(x[2], 3), d_local_cost(x[2], 4)]


def system(t, states):
    hat_X = states[:N * n]  # 估计值 hat_X
    eta = states[N * n: 2 * N * n]
    z = states[2 * N * n:]  # 拉格朗日乘子 y

    x = hat_X

    # 计算拉普拉斯项 diag(b/2) * (L⊗I) hat_X 和 (L⊗I) U1 y
    L_hatX = L @ hat_X
    L_z = L @ z


    d_hatx = (1 / xi(t)) * np.sqrt(alpha * omega) * \
             np.cos(omega * t + k * (xi(t) ** 1) * (f_vec(hat_X) - eta)) + (-a * L_hatX - z / xi(t)) * (
                         xi(t) ** 0)
    d_eta = -omega_h * eta + omega_h * np.array(f_vec(hat_X))

    d_z = b * L_hatX * (xi(t) ** 1)
    return np.concatenate([d_hatx, d_eta, d_z])

# hat_X0 = np.random.uniform(low=-1, high=5, size=N * n)
hat_X0 = np.array([-1, 0, 1, 4, 5])
eta0 = np.zeros(N * n)
z0 = np.zeros(N * n)
initial_state = np.concatenate([hat_X0, eta0, z0])

# 解常微分方程组
sol = solve_ivp(system, t_span, initial_state, method='Radau', dense_output=True)

# 提取结果
t_eval = np.linspace(0, sol.t[-1], 10000)
states = sol.sol(t_eval)
x = states[:N * n].reshape(N, -1)

# 绘制结果
fig1, ax1 = plt.subplots(figsize=(5, 4))
x_star = np.mean(c)
# sliding maximum over roughly one probing period, to show the envelope of the
# dither-induced oscillation rather than every individual excursion
ENV_WIN = max(1, int(len(t_eval) * (2 * np.pi / omega) / (t_eval[-1] - t_eval[0])))
err1 = maximum_filter1d(np.linalg.norm(x - x_star, axis=0), size=ENV_WIN)
ax1.semilogy(t_eval, err1, color='C0')
ax1.set_xlabel('Time (sec)')
ax1.set_ylabel(r'$\|\mathbf{x}(t)-\mathbf{1}_N\otimes x^*\|$', fontsize=16)
ax1.set_title('Classical ES')
ax1.grid(True)






belta = 1
def xi(t):
    return (1 + belta * t) ** (1 / v)

def system(t, states):
    hat_X = states[:N * n]  # 估计值 hat_X
    eta = states[N * n: 2 * N * n]
    z = states[2 * N * n:]  # 拉格朗日乘子 y

    x = hat_X

    # 计算拉普拉斯项 diag(b/2) * (L⊗I) hat_X 和 (L⊗I) U1 y
    L_hatX = L @ hat_X
    L_z = L @ z


    d_hatx = (1 / xi(t)) * np.sqrt(alpha * omega) * \
             np.cos(omega * t + k * (xi(t) ** 1) * (f_vec(hat_X) - eta)) + (-a * L_hatX - z / xi(t)) * (
                         xi(t) ** 0)
    d_eta = -omega_h * eta + omega_h * np.array(f_vec(hat_X))

    d_z = b * L_hatX * (xi(t) ** 1)
    return np.concatenate([d_hatx, d_eta, d_z])

# hat_X0 = np.random.uniform(low=-1, high=5, size=N * n)
hat_X0 = np.array([-1, 0, 1, 4, 5])
eta0 = np.zeros(N * n)
z0 = np.zeros(N * n)
initial_state = np.concatenate([hat_X0, eta0, z0])

# 解常微分方程组
sol = solve_ivp(system, t_span, initial_state, method='Radau', dense_output=True)

# 提取结果
t_eval = np.linspace(0, sol.t[-1], 10000)
states = sol.sol(t_eval)
x = states[:N * n].reshape(N, -1)

# 绘制结果
fig2, ax2 = plt.subplots(figsize=(5, 4))
x_star = np.mean(c)
err2 = maximum_filter1d(np.linalg.norm(x - x_star, axis=0), size=ENV_WIN)
ax2.semilogy(t_eval, err2, color='C0')
ax2.set_xlabel('Time (sec)')
ax2.set_title('Asymptotic uES')
ax2.grid(True)

# identical y-range on both panels so the two configurations are directly comparable
lo = min(err1.min(), err2.min()); hi = max(err1.max(), err2.max())
for _ax in (ax1, ax2):
    _ax.set_ylim(lo / 2, hi * 2)
for _fig, _name in ((fig1, 'out/beta=0.eps'), (fig2, 'out/beta=1.eps')):
    _fig.tight_layout(pad=0.2)
    _fig.savefig(_name, format='eps')
plt.show()
