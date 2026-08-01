import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, basinhopping
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# 设置字体参数
LATEX_FONT_SIZE = 22
plt.rcParams.update({
    'font.size': LATEX_FONT_SIZE,
    'axes.titlesize': LATEX_FONT_SIZE,
    'axes.labelsize': LATEX_FONT_SIZE,
    'xtick.labelsize': LATEX_FONT_SIZE - 1,
    'ytick.labelsize': LATEX_FONT_SIZE - 1,
    'legend.fontsize': LATEX_FONT_SIZE - 2,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
})

# 智能体参数
N = 5  # 智能体数量
c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 各智能体的本地最优值

# 目标函数定义（与原代码保持一致）
def local_cost(x_i, i):
    """智能体i的本地成本函数"""
    return (x_i - c[i]) ** 2 + np.sin((x_i - c[i]) ** 2)

def d_local_cost(x_i, i):
    """智能体i的本地成本函数导数"""
    diff = x_i - c[i]
    return 2 * diff + 2 * diff * np.cos(diff ** 2)

def global_cost(x):
    """全局成本函数：所有智能体成本函数之和"""
    if len(x) == 1:
        # 如果输入是标量，计算所有智能体在该点的总成本
        return sum(local_cost(x[0], i) for i in range(N))
    elif len(x) == N:
        # 如果输入是向量，计算每个智能体在对应点的成本之和
        return sum(local_cost(x[i], i) for i in range(N))
    else:
        raise ValueError(f"输入维度错误，期望1或{N}维，得到{len(x)}维")

def global_cost_gradient(x):
    """全局成本函数的梯度"""
    if len(x) == 1:
        # 标量情况下的梯度
        return np.array([sum(d_local_cost(x[0], i) for i in range(N))])
    elif len(x) == N:
        # 向量情况下的梯度
        return np.array([d_local_cost(x[i], i) for i in range(N)])
    else:
        raise ValueError(f"输入维度错误，期望1或{N}维，得到{len(x)}维")

# 分析解（理论最优值）
theoretical_optimum_consensus = np.mean(c)  # 一致性情况下的理论最优值
theoretical_optimum_individual = c.copy()   # 每个智能体的理论最优值

print("="*60)
print("分布式优化问题的全局成本函数最小值计算")
print("="*60)
print(f"智能体数量: {N}")
print(f"本地最优值: {c}")
print(f"理论一致性最优值: {theoretical_optimum_consensus:.4f}")
print(f"理论个体最优值: {theoretical_optimum_individual}")

# 1. 一致性优化（所有智能体状态相同）
print("\n" + "="*40)
print("1. 一致性优化（所有智能体状态相同）")
print("="*40)

def consensus_cost(x):
    """一致性成本函数（所有智能体状态相同）"""
    return global_cost([x[0]])

def consensus_gradient(x):
    """一致性成本函数梯度"""
    return global_cost_gradient([x[0]])

# 使用多种优化方法
methods = ['BFGS', 'L-BFGS-B', 'CG', 'Newton-CG']
initial_guesses = [0.0, 2.0, 4.0, np.mean(c)]

best_result = None
best_cost = float('inf')

print("\n尝试不同的优化方法和初始值：")
for method in methods:
    for x0 in initial_guesses:
        try:
            if method == 'Newton-CG':
                result = minimize(consensus_cost, [x0], method=method, 
                                jac=consensus_gradient, 
                                options={'disp': False})
            else:
                result = minimize(consensus_cost, [x0], method=method, 
                                jac=consensus_gradient, 
                                options={'disp': False})
            
            if result.success and result.fun < best_cost:
                best_cost = result.fun
                best_result = result
                
        except Exception as e:
            continue

if best_result:
    print(f"\n最佳一致性优化结果:")
    print(f"最优状态: x* = {best_result.x[0]:.6f}")
    print(f"最小成本: f(x*) = {best_result.fun:.6f}")
    print(f"理论值: x* = {theoretical_optimum_consensus:.6f}")
    print(f"理论成本: f(x*) = {consensus_cost([theoretical_optimum_consensus]):.6f}")
    print(f"优化成功: {best_result.success}")

# 2. 分布式优化（每个智能体独立）
print("\n" + "="*40)
print("2. 分布式优化（每个智能体独立优化）")
print("="*40)

# 使用差分进化算法进行全局优化
bounds = [(-2.0, 8.0) for _ in range(N)]  # 每个智能体的搜索范围
result_de = differential_evolution(global_cost, bounds, 
                                 seed=42, maxiter=1000, disp=False)

print(f"差分进化算法结果:")
print(f"最优状态: x* = {result_de.x}")
print(f"最小成本: f(x*) = {result_de.fun:.6f}")
print(f"理论最优: x* = {theoretical_optimum_individual}")
print(f"理论成本: f(x*) = {global_cost(theoretical_optimum_individual):.6f}")
print(f"优化成功: {result_de.success}")

# 3. Basin-hopping 全局优化
print("\n" + "="*40)
print("3. Basin-hopping 全局优化")
print("="*40)

initial_x = np.random.uniform(-1, 6, N)
result_bh = basinhopping(global_cost, initial_x, niter=100, disp=False)

print(f"Basin-hopping算法结果:")
print(f"最优状态: x* = {result_bh.x}")
print(f"最小成本: f(x*) = {result_bh.fun:.6f}")
print(f"优化成功: {result_bh.message}")

# 4. 可视化结果
print("\n" + "="*40)
print("4. 可视化分析")
print("="*40)

# 绘制单智能体成本函数
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

x_range = np.linspace(-1, 7, 1000)

# 子图1：各智能体的本地成本函数
for i in range(N):
    y = [local_cost(x, i) for x in x_range]
    ax1.plot(x_range, y, label=f'Agent {i+1} (c={c[i]})')
ax1.axvline(theoretical_optimum_consensus, color='red', linestyle='--', 
           label=f'Consensus opt: {theoretical_optimum_consensus:.2f}')
ax1.set_xlabel('State x')
ax1.set_ylabel('Local Cost')
ax1.set_title('Local Cost Functions')
ax1.legend()
ax1.grid(True)

# 子图2：一致性全局成本函数
global_consensus_cost = [consensus_cost([x]) for x in x_range]
ax2.plot(x_range, global_consensus_cost, 'b-', linewidth=2, label='Global Cost (Consensus)')
if best_result:
    ax2.axvline(best_result.x[0], color='red', linestyle='--', 
               label=f'Optimized: {best_result.x[0]:.3f}')
ax2.axvline(theoretical_optimum_consensus, color='green', linestyle=':', 
           label=f'Theoretical: {theoretical_optimum_consensus:.3f}')
ax2.set_xlabel('Consensus State x')
ax2.set_ylabel('Global Cost')
ax2.set_title('Global Cost Function (Consensus)')
ax2.legend()
ax2.grid(True)

# 子图3：优化结果对比
methods_names = ['Consensus\n(Theory)', 'Consensus\n(Optimized)', 
                'Distributed\n(DE)', 'Distributed\n(BH)']
costs = [
    consensus_cost([theoretical_optimum_consensus]),
    best_result.fun if best_result else float('nan'),
    result_de.fun,
    result_bh.fun
]

bars = ax3.bar(methods_names, costs, color=['green', 'blue', 'orange', 'purple'])
ax3.set_ylabel('Minimum Cost')
ax3.set_title('Optimization Results Comparison')
ax3.grid(True, axis='y')

# 在柱状图上添加数值标签
for bar, cost in zip(bars, costs):
    if not np.isnan(cost):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{cost:.3f}', ha='center', va='bottom')

# 子图4：收敛过程可视化（差分进化的收敛过程）
# 重新运行差分进化以获取收敛历史
class CallbackDE:
    def __init__(self):
        self.costs = []
    
    def __call__(self, x, convergence):
        self.costs.append(global_cost(x))

callback = CallbackDE()
result_de_detailed = differential_evolution(global_cost, bounds, 
                                          callback=callback,
                                          seed=42, maxiter=100, disp=False)

ax4.plot(callback.costs, 'b-', linewidth=2)
ax4.axhline(global_cost(theoretical_optimum_individual), color='red', 
           linestyle='--', label=f'Theoretical minimum: {global_cost(theoretical_optimum_individual):.3f}')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Best Cost')
ax4.set_title('Differential Evolution Convergence')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.savefig('out/global_optimization_analysis.eps', format='eps', bbox_inches='tight')
plt.savefig('out/global_optimization_analysis.pdf', format='pdf', bbox_inches='tight')

# 5. 总结比较
print("\n" + "="*60)
print("优化结果总结")
print("="*60)
print("方法                     | 最优值                    | 最小成本")
print("-"*60)
print(f"理论一致性最优           | {theoretical_optimum_consensus:8.4f}             | {consensus_cost([theoretical_optimum_consensus]):8.4f}")
if best_result:
    print(f"数值一致性最优           | {best_result.x[0]:8.4f}             | {best_result.fun:8.4f}")
print(f"差分进化算法             | {str(result_de.x)[:20]:<20} | {result_de.fun:8.4f}")
print(f"Basin-hopping算法        | {str(result_bh.x)[:20]:<20} | {result_bh.fun:8.4f}")
print(f"理论分布式最优           | {str(theoretical_optimum_individual):<20} | {global_cost(theoretical_optimum_individual):8.4f}")

plt.show()
print(f"\n图像已保存为 'fig/global_optimization_analysis.eps' 和 'fig/global_optimization_analysis.pdf'")
print("程序执行完成！")
