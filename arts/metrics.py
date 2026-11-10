"""Porto E1 aggregate fidelity and E2 trajectory-structure evaluation metrics."""
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from .routes import distance, shortest_distance, valid


def jsd(a, b):
    """Jensen-Shannon divergence in bits, with values between zero and one."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or np.any(a < 0) or np.any(b < 0):
        raise ValueError('JSD vectors must have matching shapes and nonnegative entries')
    if a.sum() == 0 and b.sum() == 0: return 0.
    if a.sum() == 0 or b.sum() == 0: return 1.
    p, q = a / a.sum(), b / b.sum()
    m = (p + q) / 2
    return float((np.sum(p[p > 0] * np.log2(p[p > 0] / m[p > 0])) +
                  np.sum(q[q > 0] * np.log2(q[q > 0] / m[q > 0]))) / 2)


def nmae(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-12))


def evaluate_e1(run, data):
    """Evaluate aggregate statistics and public road-network constraints."""
    state = run['state']
    target, measured = state.target, state.realized
    paths, requests = run['paths'], run['requests']
    if len(paths) != len(requests): raise ValueError('Path and OD request counts differ')
    reference_od = {q: int(np.floor(max(0., c) + 0.5)) for q, c in data.od.items()}
    produced_od = Counter(requests)
    od_error = sum(abs(produced_od[q] - reference_od.get(q, 0)) for q in set(reference_od) | set(produced_od))
    od_error /= max(sum(reference_od.values()), 1)
    connectivity = [bool(p) and all(e in data.roads for e in p) and
                    all(v in data.successors[u] for u, v in zip(p, p[1:])) for p in paths]
    endpoints = [p[0] in data.endpoints[q][0] and p[-1] in data.endpoints[q][1]
                 if p and q in data.endpoints else False for p, q in zip(paths, requests)]
    rho = run.get('max_detour')
    violations = 0 if rho is None else sum(distance(p, data) > rho * shortest_distance(q, data) + 1e-8
                                                 for p, q in zip(paths, requests) if valid(p, q, data))
    if any(not valid(p, q, data) for p, q in zip(paths, requests)):
        raise ValueError('Synthesis output contains an invalid OD route')
    rank = spearmanr(measured['flow'], target['flow']).statistic
    return {'Loss': state.loss(), 'R-NMAE': nmae(measured['region'], target['region']),
            'S-JSD': jsd(measured['scale'], target['scale']),
            'F-NMAE': nmae(measured['flow'], target['flow']),
            'F-Spearman': float(rank) if np.isfinite(rank) else 0.,
            'OD-Error': float(od_error),
            'Route-Validity': sum(connectivity) / max(len(paths), 1),
            'Endpoint-Consistency': sum(endpoints) / max(len(paths), 1),
            'Detour-Violation': violations / max(len(paths), 1)}


def evaluate_e2(paths, data, reference_file, generated_n=None, min_length=1):
    """Compare generated route distributions with fixed evaluation statistics."""
    generated_n = len(paths) if generated_n is None else int(generated_n)
    if min_length < 1: raise ValueError('Minimum route length must be positive')
    if generated_n < len(paths): raise ValueError('Generated trajectory count is inconsistent')
    legal = [p for p in paths if len(p) >= min_length and all(e in data.roads for e in p)
             and all(b in data.successors[a] for a, b in zip(p, p[1:]))]
    with np.load(Path(reference_file), allow_pickle=False) as ref:
        de, dh = ref['distance_edges'], ref['distance_hist']
        se, sh = ref['segment_edges'], ref['segment_hist']
        pairs, counts = ref['transition_pairs'], ref['transition_counts']
    dist = np.array([distance(p, data) for p in legal])
    nseg = np.array([len(p) for p in legal])
    if len(dist) and (dist.min() < de[0] or dist.max() > de[-1]):
        raise ValueError('Generated distances must be covered by evaluation bins')
    if len(nseg) and (nseg.min() < se[0] or nseg.max() > se[-1]):
        raise ValueError('Generated route lengths must be covered by evaluation bins')
    transitions = Counter((u, v) for p in legal for u, v in zip(p, p[1:]))
    reference = dict(zip(map(tuple, pairs.tolist()), counts.tolist()))
    support = sorted(set(transitions) | set(reference))
    top_ref = sorted(reference, key=lambda p: (-reference[p], p))[:100]
    top_syn = set(sorted(transitions, key=lambda p: (-transitions[p], p))[:100])
    return {'Distance-JSD': jsd(np.histogram(dist, bins=de)[0], dh),
            'SegmentCount-JSD': jsd(np.histogram(nseg, bins=se)[0], sh),
            'Transition-JSD': jsd([transitions[p] for p in support], [reference.get(p, 0) for p in support]),
            'Recall@100': len(top_syn.intersection(top_ref)) / max(len(top_ref), 1),
            'Validity': len(legal) / generated_n if generated_n else 1.}
