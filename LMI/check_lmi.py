"""
Feasibility check for the three LMIs of Proposition 1 of the manuscript.

Proposition 1 asks for P2, P3 in R^{(N-1)x(N-1)} with P3 > 0 and positive
scalars {p11, p22, delta} such that

    (i)   [ p22 I   P2 ]                (ii)  Phi_11 < 0        (iii) Phi_2 < 0
          [   *     P3 ]  >  0,

    Phi_11 = 2(beta/v) p11 - m alpha k p11 + delta M^2,
    Phi_2  = [ Phi_21   Phi_22            (p22-p11) alpha k I/2 ]
             [   *     -(P2+P2^T)/2      -alpha k P2^T/2        ]
             [   *        *              -delta I               ],
    Phi_21 = -p22 (Lc + Lc^T) + gamma (P2 Lc + Lc^T P2^T),
    Phi_22 = -p22 I + gamma Lc^T P3^T - Lc^T P2,      Lc = R^T L R.

Theorem 2 uses the same LMIs with beta/v replaced by 1/(q rho).

Two numerical points worth knowing before reading the output.

1. The LMIs are HOMOGENEOUS of degree one in (p11, p22, delta, P2, P3): scaling
   a feasible point by any positive number keeps it feasible. A plain
   feasibility problem therefore drifts to the origin, where every constraint
   holds only to solver tolerance and the answer is meaningless. We fix the
   scale by p22 = 1 and maximise a uniform margin t, so t > 0 certifies strict
   feasibility and t < 0 proves infeasibility (by homogeneity, p22 = 1 is
   without loss of generality).

2. Phi_11 < 0 forces  alpha k > 2 (beta/v) / m.  Phi_2, on the other hand,
   degrades as alpha k grows, since the (1,3) and (2,3) blocks scale with it.
   The two requirements squeeze the admissible parameters from both sides.

Run:  python3 check_lmi.py
"""

import numpy as np
import cvxpy as cp

# --------------------------------------------------------------- problem data
N = 5                                     # agents
ALPHA, K_GAIN, GAMMA = 1.0, 1.0, 1.0      # alpha, k, gamma of Section VI-A
V_PAPER, BETA_PAPER = 2.0, 1.0            # v and beta of Fig. 2(c)

# directed ring: weight balanced and strongly connected (Assumption 3)
A = np.array([[0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0]], dtype=float)
L = np.diag(A.sum(axis=1)) - A


def basis_of_ones_complement(n):
    """R in R^{n x (n-1)} with R^T r = 0 and R^T R = I, where r = 1/sqrt(n) * 1."""
    r = np.ones((n, 1)) / np.sqrt(n)
    Q, _ = np.linalg.qr(
        np.hstack([r, np.random.default_rng(0).standard_normal((n, n - 1))]))
    R = Q[:, 1:]
    assert np.allclose(R.T @ r, 0, atol=1e-12)
    assert np.allclose(R.T @ R, np.eye(n - 1), atol=1e-12)
    return R


R = basis_of_ones_complement(N)
Lc = R.T @ L @ R                          # reduced Laplacian, calligraphic L
n = N - 1


def hessian_eigen_range(radius=12.0, samples=200001):
    """
    Extremal Hessian eigenvalues of f_i(x) = |x-c_i|^2 + ln(1 + |x-c_i|^2).
    With u = |x-c_i|^2 the radial eigenvalue is 2 + (2-2u)/(1+u)^2 and the
    tangential one is 2 + 2/(1+u).
    """
    u = np.linspace(0.0, radius ** 2, samples)
    radial = 2.0 + (2.0 - 2.0 * u) / (1.0 + u) ** 2
    tangential = 2.0 + 2.0 / (1.0 + u)
    return min(radial.min(), tangential.min()), max(radial.max(), tangential.max())


M_LO, M_HI = hessian_eigen_range()         # m and M of Assumption 1


# ------------------------------------------------------------------ the LMIs
def margin(beta_over_v, gamma=GAMMA, ak=ALPHA * K_GAIN, m=M_LO, M=M_HI,
           verbose=False, label=""):
    """
    Largest uniform margin t for which the three LMIs hold with p22 = 1.
    Returns t (>0 means strictly feasible) or None if the solver failed.
    """
    p11 = cp.Variable(pos=True)
    p22 = 1.0                              # scale normalisation, WLOG
    delta = cp.Variable(pos=True)
    P2 = cp.Variable((n, n))               # not required symmetric
    P3 = cp.Variable((n, n), symmetric=True)
    t = cp.Variable()

    Phi11 = 2 * beta_over_v * p11 - m * ak * p11 + delta * M ** 2
    Phi21 = -p22 * (Lc + Lc.T) + gamma * (P2 @ Lc + Lc.T @ P2.T)
    Phi22 = -p22 * np.eye(n) + gamma * Lc.T @ P3.T - Lc.T @ P2
    Phi13 = 0.5 * (p22 - p11) * ak * np.eye(n)
    Phi23 = -0.5 * ak * P2.T

    Phi2 = cp.bmat([[Phi21,    Phi22,               Phi13],
                    [Phi22.T, -0.5 * (P2 + P2.T),   Phi23],
                    [Phi13.T,  Phi23.T,            -delta * np.eye(n)]])
    block1 = cp.bmat([[p22 * np.eye(n), P2],
                      [P2.T,            P3]])

    cons = [block1 >> t * np.eye(2 * n),
            Phi11 <= -t,
            Phi2 << -t * np.eye(3 * n),
            P3 >> t * np.eye(n),
            p11 >= t, delta >= t, t <= 1.0,
            cp.norm(P2, "fro") <= 1e4, cp.norm(P3, "fro") <= 1e4,
            p11 <= 1e4, delta <= 1e4]

    prob = cp.Problem(cp.Maximize(t), cons)
    for solver in ("CLARABEL", "SCS"):
        try:
            prob.solve(solver=solver)
            if prob.status.startswith("optimal"):
                break
        except cp.SolverError:
            continue
    if not prob.status.startswith("optimal"):
        return None

    if verbose:
        # re-check the returned point independently of the solver certificate
        b1 = np.linalg.eigvalsh(block1.value).min()
        e2 = np.linalg.eigvalsh((Phi2.value + Phi2.value.T) / 2).max()
        e3 = np.linalg.eigvalsh(P3.value).min()
        ok = (b1 > 0) and (Phi11.value < 0) and (e2 < 0) and (e3 > 0)
        print(f"--- {label} ---")
        print(f"  margin t             = {t.value:+.4e}")
        print(f"  p11 = {p11.value:.4f}  p22 = {p22:.4f}  delta = {delta.value:.4f}")
        print(f"  min eig block (i)    = {b1:+.4e}   (> 0 required)")
        print(f"  Phi_11               = {Phi11.value:+.4e}   (< 0 required)")
        print(f"  max eig Phi_2        = {e2:+.4e}   (< 0 required)")
        print(f"  min eig P3           = {e3:+.4e}   (> 0 required)")
        print(f"  -> {'STRICTLY FEASIBLE' if ok else 'INFEASIBLE'}\n")
    return t.value


def fmt(x):
    return f"{x:+8.4f}" if x is not None else "     n/a"


if __name__ == "__main__":
    print(f"cost  f_i(x) = |x-c_i|^2 + ln(1+|x-c_i|^2)")
    print(f"  m = {M_LO:.4f}  (strong convexity)   M = {M_HI:.4f}  (grad. Lipschitz)")
    print(f"  Phi_11 < 0 requires alpha*k > 2(beta/v)/m")
    print(f"  at beta=1, v=2 that is alpha*k > {2*(BETA_PAPER/V_PAPER)/M_LO:.3f}\n")

    print("=" * 66)
    print("1. the parameters actually used in Section VI-A")
    print("=" * 66)
    margin(0.0, verbose=True, label="Fig. 2(b)  beta = 0,  gamma = 1, alpha = k = 1")
    margin(BETA_PAPER / V_PAPER, verbose=True,
           label="Fig. 2(c)  beta = 1, v = 2,  gamma = 1, alpha = k = 1")

    print("=" * 66)
    print("2. dependence on gamma   (alpha = k = 1)")
    print("=" * 66)
    print(f"{'gamma':>8} | {'beta = 0':>10} {'beta = 1, v = 2':>16}")
    for g in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        print(f"{g:8g} | {fmt(margin(0.0, gamma=g))}   {fmt(margin(0.5, gamma=g))}")

    print("\n" + "=" * 66)
    print("3. dependence on v at beta = 1   (Remark 3: feasible for v large)")
    print("=" * 66)
    print(f"{'v':>6} {'beta/v':>8} | {'gamma = 1':>10} {'gamma = 0.2':>12} {'gamma = 0.05':>13}")
    for v in (2, 4, 6, 8, 10, 20, 100):
        bv = BETA_PAPER / v
        print(f"{v:6g} {bv:8.3f} | {fmt(margin(bv, gamma=1.0))}   "
              f"{fmt(margin(bv, gamma=0.2))}   {fmt(margin(bv, gamma=0.05))}")

    print("\nA positive margin certifies that the three LMIs admit a strictly")
    print("feasible (p11, p22, delta, P2, P3); a negative one proves they do not.")
