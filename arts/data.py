"""Porto road-network inputs, aggregate statistics, and calibrated OD requests."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np

@dataclass
class PortoData:
    roads: dict
    successors: dict
    endpoints: dict
    od: dict
    targets: dict
    scale_edges: np.ndarray
    library: dict = field(default_factory=dict)
    road_index: dict = field(init=False)
    shortest_lengths: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.road_index = {e: i for i, e in enumerate(sorted(self.roads))}
        if not self.roads or set(self.roads) != set(self.successors):
            raise ValueError('Road segments and successor lists must share the same IDs')
        if any(v not in self.roads for nxt in self.successors.values() for v in nxt):
            raise ValueError('Successor relation contains unknown road segments')
        if any(not np.isfinite(d) or d <= 0 or not 0 <= r < len(self.targets['region'])
               for d, r in self.roads.values()):
            raise ValueError('Invalid road length or Region index')
        if len(self.targets['flow']) != len(self.roads) or len(self.scale_edges) != len(self.targets['scale']) + 1:
            raise ValueError('Aggregate vector dimensions do not match road and Scale configurations')
        if not np.all(np.isfinite(self.scale_edges)) or not np.all(np.diff(self.scale_edges) > 0):
            raise ValueError('Scale bin boundaries must be finite and strictly increasing')
        if any(v.ndim != 1 or not len(v) or not np.all(np.isfinite(v)) for v in self.targets.values()):
            raise ValueError('Aggregate statistics must be finite nonempty vectors')
        if any(not np.isfinite(v) for v in self.od.values()):
            raise ValueError('OD counts must be finite')
        if any(not starts or not ends or not (starts | ends) <= self.roads.keys()
               for starts, ends in self.endpoints.values()):
            raise ValueError('OD endpoint sets must contain valid road segments')


def load_porto(folder):
    """Load the public road network, OD demand, aggregate targets, and route libraries."""
    p = Path(folder)
    net = json.loads((p / 'network.json').read_text(encoding='utf-8'))
    roads = {int(x['id']): (float(x['length_m']), int(x['region'])) for x in net['roads']}
    successors = {int(k): set(map(int, v)) for k, v in net['successors'].items()}
    req = json.loads((p / 'od.json').read_text(encoding='utf-8'))
    od, endpoints = {}, {}
    for x in req:
        q = (str(x['origin']), str(x['destination']))
        if q in od: raise ValueError(f'Duplicate OD entry: {q}')
        od[q] = float(x['count'])
        endpoints[q] = (set(map(int, x['starts'])), set(map(int, x['ends'])))
    with np.load(p / 'targets.npz', allow_pickle=False) as f:
        targets = {k: np.asarray(f[k], dtype=float) for k in ('region', 'scale', 'flow')}
        edges = np.asarray(f['scale_edges'], dtype=float)
    library = {}
    if (p / 'library.json').is_file():
        for x in json.loads((p / 'library.json').read_text(encoding='utf-8')):
            q = (str(x['origin']), str(x['destination']))
            if q in library: raise ValueError(f'Duplicate route-library entry: {q}')
            library[q] = [tuple(map(int, path)) for path in x['paths']]
    return PortoData(roads, successors, endpoints, od, targets, edges, library)


def calibrated_requests(data, seed=42):
    """Calibrate OD counts by half-up rounding and permute the feasible requests."""
    requests, excluded = [], {}
    for q, count in sorted(data.od.items()):
        n = int(np.floor(max(0., count) + 0.5))
        if not n: continue
        if not data.library.get(q):
            excluded[q] = n
            continue
        requests.extend([q] * n)
    order = np.random.default_rng(seed).permutation(len(requests))
    return [requests[i] for i in order], excluded
