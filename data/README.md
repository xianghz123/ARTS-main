# Porto Data Preparation

The ARTS Porto E1/E2 experiments use a directed road-segment network, calibrated OD demand, released mobility statistics, candidate route libraries, and evaluation reference statistics. Place the following five files under `data/porto/`:

```text
data/porto/
├── network.json
├── od.json
├── targets.npz
├── library.json
└── e2_reference.npz
```

All road IDs, route sequences, Region indices, distance units, and reference statistics must use the same Porto preprocessing convention. The scripts load these normalized files directly.

## 1. `network.json`: Road Network

The file defines road segments and directed legal successors:

```json
{
  "roads": [
    {"id": 0, "length_m": 24.0, "region": 0},
    {"id": 1, "length_m": 32.0, "region": 0}
  ],
  "successors": {"0": [1], "1": []}
}
```

| Field | Description |
| --- | --- |
| `id` | Unique integer road-segment identifier |
| `length_m` | Positive road-segment length in meters |
| `region` | Integer Region index, starting at zero |
| `successors` | Directed legal segment transitions keyed by road ID |

Every road segment must have a successor entry, including terminal segments with an empty successor list. The `flow` target in `targets.npz` uses **ascending road IDs** as its coordinate order. The JSON block illustrates the input schema.

## 2. `od.json`: OD Demand

Each record describes an OD pair, its released trip count, and its permitted start and destination road segments:

```json
[
  {"origin": "o0", "destination": "d0", "count": 2, "starts": [0], "ends": [1]}
]
```

| Field | Description |
| --- | --- |
| `origin` | Origin identifier |
| `destination` | Destination identifier |
| `count` | Nonnegative OD demand |
| `starts` | Road segments associated with the origin |
| `ends` | Road segments associated with the destination |

OD counts are calibrated to nonnegative integer demands using half-up rounding. The Porto experiment contains **9,968 OD requests** after calibration. Every positive-demand OD pair must have a candidate route in `library.json`. The five random seeds `42`–`46` determine the request permutation.

## 3. `targets.npz`: Aggregate Mobility Statistics

This NumPy archive contains four arrays:

| Array | Definition |
| --- | --- |
| `region` | Region coverage counts, indexed by contiguous Region IDs |
| `scale` | Trip counts in the Scale distance bins |
| `flow` | Road-segment traversal counts, indexed by ascending road ID |
| `scale_edges` | Strictly increasing boundaries of the Scale bins |

The number of Scale boundaries equals `len(scale) + 1`; the number of Flow entries equals the number of road segments. For a route, Region coverage counts each visited Region once, Scale assigns the route distance to one bin, and Flow counts road-segment traversals. Route distance is the sum of `length_m` along the path.

ARTS calibrates the published targets before synthesis: Region counts are clipped to the range `[0, N]`, Scale counts are projected onto the nonnegative simplex with total mass `N`, and Flow counts are projected to nonnegative values. E1 uses the calibrated targets in the unified normalized discrepancy objective.

## 4. `library.json`: Candidate Route Libraries

Each record contains the feasible route alternatives for one OD pair:

```json
[
  {"origin": "o0", "destination": "d0", "paths": [[0, 1]]}
]
```

Each route is an ordered sequence of road-segment IDs. Candidate paths satisfy endpoint consistency, directed road connectivity, looplessness, and the bounded-detour condition

\[
\ell(\tau) \leq \rho_{\max}\,\ell_{\mathrm{sp}}(q), \qquad \rho_{\max}=3.0.
\]

The Porto route-library configuration uses **at most 10 candidate routes per OD** and a pairwise road-segment Jaccard overlap threshold of **0.8**. Each library includes a shortest feasible OD route. E1 reads `library.json` to provide the same candidates to all five methods. The `arts/routes.py` module also provides route-library construction and validation functions.

## 5. `e2_reference.npz`: Structural Evaluation Reference

This archive stores the reference distributions used in E2:

| Array | Definition |
| --- | --- |
| `distance_edges` | Distance-histogram bin boundaries |
| `distance_hist` | Reference counts by distance bin |
| `segment_edges` | Segment-count histogram boundaries |
| `segment_hist` | Reference counts by segment-count bin |
| `transition_pairs` | Array of directed road-segment pairs with shape `(M, 2)` |
| `transition_counts` | Reference frequency of each transition, with shape `(M,)` |

The histogram boundary arrays contain one more value than their corresponding count arrays. `transition_pairs` and `transition_counts` have equal first dimensions. E2 compares the saved E1 trajectories with these distributions using Distance-JSD, SegmentCount-JSD, Transition-JSD, and Recall@100, and computes the valid-trajectory ratio.

This reference is supplied to **`run_e2.py`**; it is separate from the four files used for E1 synthesis and route selection. E2 evaluates road-network connectivity for all nonempty routes by default (`min_length_e2: 1`); the `--min-length` option applies an explicit segment-count threshold when required by a dataset protocol. E2 uses the same distance units, bin edges, road IDs, and structural validity rules as the Porto evaluation protocol.

## Execution

From the repository root, after preparing the five files:

```bash
python scripts/run_e1.py
python scripts/run_e2.py
```

The default commands run all five methods with seeds `42, 43, 44, 45, 46`. Single-seed evaluation is available through `--seed 42`. The generated result files and multi-seed summaries are described in the repository [README](../README.md).

