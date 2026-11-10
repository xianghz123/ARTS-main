"""Porto E1: OD-constrained trajectory synthesis and aggregate-statistic evaluation."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arts.data import calibrated_requests, load_porto
from arts.metrics import evaluate_e1
from arts.routes import validate_library
from arts.synthesis import METHODS, synthesize

INPUTS = ('network.json', 'od.json', 'targets.npz', 'library.json')
COLUMNS = ('Loss', 'R-NMAE', 'S-JSD', 'F-NMAE', 'F-Spearman', 'OD-Error',
           'Route-Validity', 'Endpoint-Consistency', 'Detour-Violation')
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_trajectory(path, record):
    """Store a deterministic compressed trajectory record."""
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8') as stream:
                json.dump(record, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def summarize(rows, folder, seeds, methods):
    if len(seeds) < 2: return
    write_csv(folder / 'summary_all_seeds.csv', rows, ('Seed', 'Method', *COLUMNS))
    averages, deviations = [], []
    for method in methods:
        group = [r for r in rows if r['Method'] == method]
        averages.append({'Method': method, 'Seeds': len(group), **{k: mean(r[k] for r in group) for k in COLUMNS}})
        deviations.append({'Method': method, 'Seeds': len(group), **{k: stdev(r[k] for r in group) for k in COLUMNS}})
    write_csv(folder / 'summary_mean.csv', averages, ('Method', 'Seeds', *COLUMNS))
    write_csv(folder / 'summary_std.csv', deviations, ('Method', 'Seeds', *COLUMNS))


def main():
    parser = argparse.ArgumentParser(description='ARTS Porto E1: five-method synthesis and aggregate evaluation')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/porto')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/e1/porto')
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument('--seed', type=int, default=None)
    seeds.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--expected-n', type=int, default=9968)
    parser.add_argument('--max-detour', type=float, default=3.0)
    parser.add_argument('--refine-rounds', type=int, default=100)
    parser.add_argument('--eta', type=float, default=1e-15)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    selected_seeds = args.seeds if args.seeds is not None else ([args.seed] if args.seed is not None else list(DEFAULT_SEEDS))
    if len(selected_seeds) != len(set(selected_seeds)): parser.error('Seed values must be unique')
    if args.expected_n < 1 or args.max_detour < 1 or not math.isfinite(args.max_detour):
        parser.error('Expected trajectory count and detour constraint must be positive')
    if args.refine_rounds < 0 or args.eta <= 0 or not math.isfinite(args.eta):
        parser.error('Refinement parameters must be valid finite values')
    inputs = {name: args.data_dir / name for name in INPUTS}
    for path in inputs.values():
        if not path.is_file(): parser.error(f'Required Porto input: {path}')
    data = load_porto(args.data_dir)
    validate_library(data, args.max_detour)
    for seed in selected_seeds:
        requests, excluded = calibrated_requests(data, seed)
        if excluded: parser.error(f'OD candidate coverage incomplete for seed {seed}: {len(excluded)} OD pairs')
        if len(requests) != args.expected_n:
            parser.error(f'OD demand for seed {seed}: {len(requests)}; expected: {args.expected_n}')
    names = {method: f'{method.lower().replace("-", "_")}.json.gz' for method in args.methods}
    for seed in selected_seeds:
        folder = args.output_dir / f'seed_{seed}'
        targets = [folder / name for name in (*names.values(), 'manifest.json', 'summary.csv')]
        if not args.overwrite and any(path.exists() for path in targets):
            parser.error(f'Output directory already contains E1 artifacts: {folder}; use --overwrite')
    if not args.overwrite and len(selected_seeds) > 1 and any((args.output_dir / name).exists()
            for name in ('summary_all_seeds.csv', 'summary_mean.csv', 'summary_std.csv')):
        parser.error(f'Aggregate E1 summaries already exist: {args.output_dir}; use --overwrite')
    input_hashes = {name: sha256(path) for name, path in inputs.items()}
    all_rows = []
    for seed in selected_seeds:
        out = args.output_dir / f'seed_{seed}'
        out.mkdir(parents=True, exist_ok=True)
        reference_counts = Counter({q: int(math.floor(max(0., c) + 0.5)) for q, c in data.od.items()})
        rows, artifacts = [], {}
        for method in args.methods:
            run = synthesize(data, method=method, seed=seed, refine_rounds=args.refine_rounds,
                             eta=args.eta, max_detour=args.max_detour)
            metrics = evaluate_e1(run, data)
            if len(run['paths']) != args.expected_n or Counter(run['requests']) != reference_counts:
                raise RuntimeError(f'{method}: OD allocation constraint failed')
            if any(not math.isfinite(float(metrics[k])) for k in COLUMNS):
                raise RuntimeError(f'{method}: Nonfinite evaluation metric')
            for key, value in {'OD-Error': 0., 'Route-Validity': 1., 'Endpoint-Consistency': 1.,
                               'Detour-Violation': 0.}.items():
                if abs(metrics[key] - value) > 1e-10:
                    raise RuntimeError(f'{method}: {key} constraint failed')
            record = {'method': method, 'seed': seed, 'paths': [list(p) for p in run['paths']],
                      'requests': [list(q) for q in run['requests']], 'choices': run['choices'],
                      'generated_n': len(run['paths']), 'initial_loss': run['initial_loss'],
                      'final_loss': run['final_loss'], 'swaps': run['swaps'],
                      'sweeps': run.get('sweeps', 0), 'metrics': metrics}
            artifact = out / names[method]
            write_trajectory(artifact, record)
            artifacts[method] = {'file': artifact.name, 'sha256': sha256(artifact)}
            rows.append({'Method': method, **metrics})
            all_rows.append({'Seed': seed, 'Method': method, **metrics})
            print(f'[E1] seed={seed} method={method} N={len(run["paths"])} '
                  f'Loss={metrics["Loss"]:.6f} Status=PASS', flush=True)
        write_csv(out / 'summary.csv', rows, ('Method', *COLUMNS))
        parameters = {'seed': seed, 'expected_n': args.expected_n, 'max_detour': args.max_detour,
                      'refine_rounds': args.refine_rounds, 'eta': args.eta, 'delta': 1e-12,
                      'candidate_limit': 10, 'diversity_threshold': 0.8}
        manifest = {'dataset': 'Porto', 'experiment': 'E1', 'params': parameters,
                    'inputs_sha256': input_hashes, 'artifacts': artifacts}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summarize(all_rows, args.output_dir, selected_seeds, args.methods)
    print(f'E1 COMPLETE: seeds={selected_seeds} methods={len(args.methods)} output={args.output_dir}')


if __name__ == '__main__': main()
