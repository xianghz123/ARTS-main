"""Same-input baselines and ARTS dynamic residual-guided route realization."""
import numpy as np
from .data import calibrated_requests
from .routes import distance, validate_library
from .statistics import AggregateState, contribution

METHODS = ('SP', 'Lib-Uniform', 'Static-Best', 'Static-Mix', 'ARTS')


def synthesize(data, method='ARTS', seed=42, refine_rounds=100, eta=1e-15, max_detour=3.):
    """Realize OD requests and apply immediate same-OD replacement refinement."""
    if method not in METHODS: raise ValueError(f'Unknown method: {method}')
    if refine_rounds < 0 or eta <= 0: raise ValueError('Invalid refinement parameters')
    validate_library(data, max_detour)
    requests, excluded = calibrated_requests(data, seed)
    if excluded: raise ValueError(f'Positive-demand OD requests have no candidate library: {excluded}')
    state = AggregateState(data.targets, len(requests))
    rng = np.random.default_rng(seed)
    feats = {q: [contribution(path, data) for path in paths] for q, paths in data.library.items()}
    lengths = {q: [distance(path, data) for path in paths] for q, paths in data.library.items()}
    initial_scores = {q: np.array([state.gain(f) for f in fs]) for q, fs in feats.items()}
    mix = {}
    for q, scores in initial_scores.items():
        weights = np.maximum(scores, 0.)
        mix[q] = weights / weights.sum() if weights.sum() > 0 else np.full(len(scores), 1. / len(scores))
    assignments = []
    for q in requests:
        if method == 'SP':
            choice = min(range(len(lengths[q])), key=lambda j: (lengths[q][j], j))
        elif method == 'Lib-Uniform':
            choice = int(rng.integers(len(feats[q])))
        elif method == 'Static-Best':
            choice = min(range(len(feats[q])), key=lambda j: (-initial_scores[q][j], lengths[q][j], j))
        elif method == 'Static-Mix':
            choice = int(rng.choice(len(feats[q]), p=mix[q]))
        else:
            scores = [state.gain(f) for f in feats[q]]
            choice = min(range(len(scores)), key=lambda j: (-scores[j], lengths[q][j], j))
        assignments.append(choice)
        state.update(feats[q][choice])
    initial_loss, swaps, sweeps = state.loss(), 0, 0
    if method == 'ARTS':
        for _ in range(refine_rounds):
            accepted = 0
            for i, q in enumerate(requests):
                current = assignments[i]
                old = feats[q][current]
                best, best_gain = current, eta
                for j, new in enumerate(feats[q]):
                    if j == current: continue
                    gain = state.gain(new, old)
                    if gain > best_gain:
                        best, best_gain = j, gain
                if best == current: continue
                state.update(old, -1)
                state.update(feats[q][best])
                assignments[i] = best
                accepted += 1
            sweeps += 1
            swaps += accepted
            if not accepted: break
    paths = [data.library[q][j] for q, j in zip(requests, assignments)]
    return {'method': method, 'seed': seed, 'paths': paths, 'requests': requests,
            'choices': assignments, 'excluded_od': excluded, 'state': state,
            'initial_loss': initial_loss, 'final_loss': state.loss(), 'swaps': swaps,
            'sweeps': sweeps, 'max_detour': max_detour}
