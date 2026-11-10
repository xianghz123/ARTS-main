"""Calibrated mobility statistics, additive trajectory contributions, and exact gains."""
from collections import Counter
import numpy as np

STATISTICS = ('region', 'scale', 'flow')


def simplex_projection(x, mass):
    """Project a vector onto the nonnegative simplex with total mass M."""
    x = np.asarray(x, dtype=float)
    if mass < 0 or not len(x): raise ValueError('Invalid simplex configuration')
    if mass == 0: return np.zeros_like(x)
    u = np.sort(x)[::-1]
    idx = np.arange(1, len(u) + 1)
    cs = np.cumsum(u)
    active = np.flatnonzero(u - (cs - mass) / idx > 0)
    theta = (cs[active[-1]] - mass) / (active[-1] + 1)
    return np.maximum(x - theta, 0.)


def calibrate(targets, n):
    """Apply statistic-specific Euclidean projections to published targets."""
    return {'region': np.clip(targets['region'], 0., n),
            'scale': simplex_projection(targets['scale'], n),
            'flow': np.maximum(targets['flow'], 0.)}


def contribution(path, data):
    """Return Region coverage, Scale-bin membership, and segment traversal counts."""
    length = sum(data.roads[e][0] for e in path)
    b = int(np.searchsorted(data.scale_edges, length, side='right') - 1)
    if length == data.scale_edges[-1]: b = len(data.scale_edges) - 2
    if not 0 <= b < len(data.targets['scale']):
        raise ValueError(f'Route length {length} is outside the configured Scale bins')
    return {'region': {data.roads[e][1]: 1 for e in path},
            'scale': {b: 1},
            'flow': dict(Counter(data.road_index[e] for e in path))}


class AggregateState:
    """Maintain the normalized discrepancy and sparse residual updates."""
    def __init__(self, targets, n, weights=None, delta=1e-12):
        if delta <= 0: raise ValueError('delta must be positive')
        self.target = calibrate(targets, n)
        self.realized = {v: np.zeros_like(self.target[v]) for v in STATISTICS}
        self.residual = {v: self.target[v].copy() for v in STATISTICS}
        self.denom = {v: float(self.target[v] @ self.target[v] + delta) for v in STATISTICS}
        raw = {v: 1. for v in STATISTICS} if weights is None else {v: float(weights.get(v, 0.)) for v in STATISTICS}
        if any(w < 0 or not np.isfinite(w) for w in raw.values()) or sum(raw.values()) <= 0:
            raise ValueError('Aggregate weights must be nonnegative with positive total mass')
        self.weights = {v: w / sum(raw.values()) for v, w in raw.items()}

    def loss(self):
        return sum(self.weights[v] * float(r @ r) / self.denom[v]
                   for v, r in self.residual.items())

    def gain(self, new, old=None):
        """Compute the exact reduction in objective caused by a route insertion or replacement."""
        total = 0.
        for v in STATISTICS:
            diff = Counter(new[v])
            if old is not None: diff.subtract(old[v])
            total += self.weights[v] * sum(2 * self.residual[v][i] * d - d * d
                                           for i, d in diff.items() if d) / self.denom[v]
        return float(total)

    def update(self, feat, sign=1):
        for v in STATISTICS:
            for i, count in feat[v].items():
                self.realized[v][i] += sign * count
                self.residual[v][i] -= sign * count
