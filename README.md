# ARTS: Aggregate-Guided Road-Network Trajectory Synthesis

This repository provides the **Porto E1/E2 experiment pipeline** for ARTS. It generates road-network trajectories from a public directed road network and released **Origin-Destination (OD), Region, Scale, and Flow statistics**. The OD demand and road-network constraints are preserved during synthesis, while auxiliary aggregate statistics guide route selection and same-OD refinement.

The pipeline evaluates five methods under common OD requests and route libraries: **SP**, **Lib-Uniform**, **Static-Best**, **Static-Mix**, and **ARTS**.

## Repository Structure

```text
ARTS-main/
├── arts/
│   ├── __init__.py
│   ├── data.py
│   ├── statistics.py
│   ├── routes.py
│   ├── synthesis.py
│   └── metrics.py
├── scripts/
│   ├── run_e1.py
│   └── run_e2.py
├── data/
│   ├── README.md
│   └── porto/
│       ├── network.json
│       ├── od.json
│       ├── targets.npz
│       ├── library.json
│       └── e2_reference.npz
├── config.yaml
├── README.md
└── requirements.txt
```

## Environment

Use Python 3.10 or newer. Install the required packages from the repository root:

```bash
python -m pip install -r requirements.txt
```

Prepare the Porto input files according to [data/README.md](data/README.md). The four E1 inputs are `network.json`, `od.json`, `targets.npz`, and `library.json`; `e2_reference.npz` supplies evaluation statistics for E2.

## Experimental Configuration

| Setting | Value |
| --- | --- |
| Dataset | Porto |
| Random seeds | 42, 43, 44, 45, 46 |
| OD requests per run | 9,968 |
| Candidate routes per OD, `K` | 10 maximum |
| Detour threshold, `rho_max` | 3.0 |
| Route-overlap threshold, `theta_E` | 0.8 |
| Maximum refinement sweeps, `J_max` | 100 |
| Refinement acceptance threshold, `eta` | `1e-15` |
| Discrepancy stabilizer, `delta` | `1e-12` |
| Auxiliary-statistic weights | Equal normalized weights |
| E2 minimum segment count | 1 |

`config.yaml` records the experiment settings. The run scripts receive their runtime parameters through command-line options; execute `python scripts/run_e1.py --help` or `python scripts/run_e2.py --help` for the available arguments. Candidate-library settings and the discrepancy stabilizer are defined in the corresponding `arts/` modules.

## E1: Aggregate Fidelity

Run all five methods for the five seeds:

```bash
python scripts/run_e1.py
```

Run one seed or one method:

```bash
python scripts/run_e1.py --seed 42
python scripts/run_e1.py --seed 42 --methods ARTS
```

E1 calibrates OD requests, validates the route libraries, synthesizes trajectories, and evaluates **R-NMAE**, **S-JSD**, **F-NMAE**, **F-Spearman**, and the normalized **Loss**. It also checks OD preservation, route validity, endpoint consistency, and bounded detours.

Results are written to:

```text
outputs/e1/porto/
├── seed_42/
│   ├── sp.json.gz
│   ├── lib_uniform.json.gz
│   ├── static_best.json.gz
│   ├── static_mix.json.gz
│   ├── arts.json.gz
│   ├── summary.csv
│   └── manifest.json
├── seed_43/
├── seed_44/
├── seed_45/
├── seed_46/
├── summary_all_seeds.csv
├── summary_mean.csv
└── summary_std.csv
```

Each trajectory archive stores the generated road-segment sequences, OD requests, candidate selections, and evaluation results. `manifest.json` records the experiment parameters and SHA-256 hashes of the E1 inputs and generated archives. `summary_mean.csv` and `summary_std.csv` provide five-seed aggregate results.

## E2: Structural Fidelity

After E1, run the structural evaluation on the same synthetic trajectories:

```bash
python scripts/run_e2.py
```

For a single seed or method:

```bash
python scripts/run_e2.py --seed 42
python scripts/run_e2.py --seed 42 --methods ARTS
```

E2 reads the E1 trajectory archives and uses `data/porto/e2_reference.npz` to evaluate **Distance-JSD**, **SegmentCount-JSD**, **Transition-JSD**, **Recall@100**, and **Validity**. It checks the SHA-256 manifest and the correspondence between OD requests, selected routes, and generated trajectories. E2 evaluates the saved E1 outputs directly without generating new trajectories.

Results are written to:

```text
outputs/e2/porto/
├── seed_42/
│   ├── summary.csv
│   └── metrics.json
├── seed_43/
├── seed_44/
├── seed_45/
├── seed_46/
├── summary_all_seeds.csv
├── summary_mean.csv
└── summary_std.csv
```

## Reproducibility

E1 and E2 share the same normalized Porto road identifiers, OD definitions, and candidate route libraries. All five methods use the same seed-specific request order. E2 reference statistics are used exclusively for evaluation and do not participate in E1 route selection.

To rerun an existing experiment and replace its output files, append `--overwrite` to the corresponding command. The run records, SHA-256 manifests, and per-seed summaries support deterministic execution and traceable evaluation.

The code in this repository covers **E1 aggregate fidelity and E2 structural fidelity** with the five same-input methods. The reported results are computed from the Porto input files supplied under `data/porto/`.
