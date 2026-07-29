# Distributed Time-Varying Optimization via Unbiased Extremum Seeking — Simulation Code

This repository contains the Python code that reproduces all numerical results
in the paper:

> X. Li, X. Yang, E. Fridman, M. Diagne, and J. Sun,
> "Distributed Time-Varying Optimization via Unbiased Extremum Seeking,"
> submitted to *IEEE Transactions on Control of Network Systems* (TCNS).

The framework is a distributed, continuous-time, **gradient-free** scheme based
on **unbiased extremum seeking (uES)**: `N` agents cooperatively track the
optimum `x*(t)` of a time-varying sum cost `f(x, ζ(t)) = Σ fᵢ(x, ζ(t))` using
only real-time function measurements, over a weight-balanced, strongly connected
directed graph. A time-scaling ("chirpy") probing mechanism unifies
asymptotic, exponential, and prescribed-time convergence.

## Requirements

- Python ≥ 3.9
- `numpy`, `scipy`, `matplotlib` (figures)
- `cvxpy` with a SDP solver such as CLARABEL or SCS (only for `LMI/`)

```bash
pip install -r requirements.txt
```

## Layout

```
.
├── *.py          # figure-generating scripts (see the map below)
├── out/          # generated figures (.eps / .pdf), tracked for convenience
└── LMI/          # LMI feasibility checks for Proposition 1 + parameter search
```

## Script → figure map

| script | outputs (`out/`) | figure in the paper |
|---|---|---|
| `fig2_beta_3d.py` | `beta3d=0.eps`, `beta3d=1.eps` | **Fig. 2** — 3-D (`d=3`) constant-frequency example; `β=0` (biased ES) vs `β=1` (unbiased) |
| `fig4_chirpy_invariant.py` | `chirpy_{asy,exp,pt}_invariant.eps` | **Fig. 3** — chirpy probing, time-invariant extremum |
| `fig5_chirpy_varying.py` | `chirpy_{asy,exp,pt}_varying.eps` | **Fig. 4** — chirpy probing, time-varying extremum |
| `fig3_beta.py` | `beta=0.eps`, `beta=1.eps` | superseded scalar (`d=1`) version of Fig. 2, kept for reference |

Fig. 1 (block diagram) and the topology panel of Fig. 2 are drawn separately and
are not produced here.

## Reproducing the figures

```bash
pip install -r requirements.txt
python fig2_beta_3d.py          # -> out/beta3d=0.eps, out/beta3d=1.eps
python fig4_chirpy_invariant.py # -> out/chirpy_*_invariant.eps
python fig5_chirpy_varying.py   # -> out/chirpy_*_varying.eps
```

Each script writes its figures into `out/`.

## LMI feasibility (`LMI/`)

`LMI/check_lmi.py` verifies the three LMIs of Proposition 1 as a semidefinite
program and reports a feasibility margin for the parameter set used in the
paper, together with sweeps used to select those parameters. See
[`LMI/README.md`](LMI/README.md) for details. Requires `cvxpy`.

## Notes on parameter choices

The probing frequencies, gains, and time-scaling parameters used in the scripts
are chosen to be **strictly LMI-feasible** for Proposition 1. The rationale
behind these choices (feasibility margins, the role of `β/v`, and the
frequency-selection rule that avoids resonant sidebands when `d > 1`) is
documented in the READMEs and in the comments of the individual scripts.

## Citation

If you use this code, please cite the paper above. A `CITATION.cff` /
BibTeX entry will be added upon acceptance.

## License

Released under the [MIT License](LICENSE).
