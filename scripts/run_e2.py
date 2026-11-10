"""Porto E2: structural-fidelity evaluation of E1 synthetic road-network trajectories."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arts.data import calibrated_requests, load_porto
from arts.metrics import evaluate_e2
from arts.synthesis import METHODS

COLUMNS = ('Distance-JSD', 'SegmentCount-JSD', 'Transition-JSD', 'Recall@100', 'Validity')
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


def summarize(rows, folder, seeds, methods):
    if len(seeds) < 2: return
    write_csv(folder / 'summary_all_seeds.csv', rows, ('Seed', 'Method', *COLUMNS))
    averages, deviations = [], []
    for method in methods:
        group = [r for r in rows if r['Method'] == method]
        if len(group) != len(seeds): raise RuntimeError(f'Incomplete seed records for {method}')
        averages.append({'Method': method, 'Seeds': len(group), **{k: mean(r[k] for r in group) for k in COLUMNS}})
        deviations.append({'Method': method, 'Seeds': len(group), **{k: stdev(r[k] for r in group) for k in COLUMNS}})
    write_csv(folder / 'summary_mean.csv', averages, ('Method', 'Seeds', *COLUMNS))
    write_csv(folder / 'summary_std.csv', deviations, ('Method', 'Seeds', *COLUMNS))


def main():
    parser = argparse.ArgumentParser(description='ARTS Porto E2: E1 trajectory structure and transition statistics')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/porto')
    parser.add_argument('--e1-dir', type=Path, default=ROOT / 'outputs/e1/porto')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/e2/porto')
    parser.add_argument('--reference', type=Path, default=None)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument('--seed', type=int, default=None)
    seeds.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--expected-n', type=int, default=9968)
    parser.add_argument('--min-length', type=int, default=1)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    selected_seeds = args.seeds if args.seeds is not None else ([args.seed] if args.seed is not None else list(DEFAULT_SEEDS))
    if len(selected_seeds) != len(set(selected_seeds)): parser.error('Seed values must be unique')
    if args.expected_n < 1 or args.min_length < 1: parser.error('Invalid evaluation configuration')
    reference = args.reference or args.data_dir / 'e2_reference.npz'
    if not reference.is_file(): parser.error(f'E2 reference file: {reference}')
    for seed in selected_seeds:
        folder = args.e1_dir / f'seed_{seed}'
        if not (folder / 'manifest.json').is_file(): parser.error(f'E1 manifest file: {folder / "manifest.json"}')
        output = args.output_dir / f'seed_{seed}'
        if not args.overwrite and any((output / name).exists() for name in ('summary.csv', 'metrics.json')):
            parser.error(f'E2 results already exist: {output}; use --overwrite')
    if not args.overwrite and len(selected_seeds) > 1 and any((args.output_dir / name).exists()
            for name in ('summary_all_seeds.csv', 'summary_mean.csv', 'summary_std.csv')):
        parser.error(f'Aggregate E2 summaries already exist: {args.output_dir}; use --overwrite')
    data = load_porto(args.data_dir)
    ref_hash = sha256(reference)
    all_rows, selected_methods = [], None
    for seed in selected_seeds:
        source = args.e1_dir / f'seed_{seed}'
        manifest_file = source / 'manifest.json'
        manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        params = manifest['params']
        if manifest['dataset'] != 'Porto' or params['seed'] != seed or params['expected_n'] != args.expected_n:
            parser.error(f'E1 configuration mismatch: seed {seed}')
        for name, digest in manifest['inputs_sha256'].items():
            path = args.data_dir / name
            if not path.is_file() or sha256(path) != digest:
                parser.error(f'E1 input integrity check failed: {path}')
        expected_requests, excluded = calibrated_requests(data, seed)
        if excluded or len(expected_requests) != args.expected_n:
            parser.error(f'OD request consistency check failed: seed {seed}')
        methods = args.methods if args.methods is not None else [m for m in METHODS if m in manifest['artifacts']]
        if selected_methods is None: selected_methods = methods
        if methods != selected_methods: parser.error('The evaluated method set must match across seeds')
        out = args.output_dir / f'seed_{seed}'
        out.mkdir(parents=True, exist_ok=True)
        rows, results = [], {}
        for method in methods:
            if method not in manifest['artifacts']: parser.error(f'Expected E1 method: {method}')
            entry = manifest['artifacts'][method]
            artifact = source / entry['file']
            if not artifact.is_file() or sha256(artifact) != entry['sha256']:
                parser.error(f'E1 artifact integrity check failed: {artifact}')
            with gzip.open(artifact, 'rt', encoding='utf-8') as stream: record = json.load(stream)
            paths = [tuple(map(int, path)) for path in record['paths']]
            requests = [tuple(q) for q in record['requests']]
            choices = record['choices']
            if record['method'] != method or record['seed'] != seed or record['generated_n'] != args.expected_n:
                parser.error(f'E1 trajectory metadata mismatch: {method}, seed {seed}')
            if len(paths) != args.expected_n or requests != expected_requests or len(choices) != len(paths):
                parser.error(f'E1 trajectory allocation mismatch: {method}, seed {seed}')
            if any(q not in data.library or not isinstance(j, int) or not 0 <= j < len(data.library[q])
                   or tuple(data.library[q][j]) != path for q, j, path in zip(requests, choices, paths)):
                parser.error(f'E1 route-library correspondence check failed: {method}, seed {seed}')
            metrics = evaluate_e2(paths, data, reference, generated_n=len(paths), min_length=args.min_length)
            if any(not math.isfinite(float(metrics[k])) for k in COLUMNS):
                raise RuntimeError(f'{method}: Nonfinite evaluation metric')
            if abs(metrics['Validity'] - 1.0) > 1e-12:
                raise RuntimeError(f'{method}: Route validity constraint failed')
            rows.append({'Method': method, **metrics})
            all_rows.append({'Seed': seed, 'Method': method, **metrics})
            results[method] = metrics
            print(f'[E2] seed={seed} method={method} Transition-JSD={metrics["Transition-JSD"]:.6f} '
                  f'Status=PASS', flush=True)
        write_csv(out / 'summary.csv', rows, ('Method', *COLUMNS))
        report = {'dataset': 'Porto', 'experiment': 'E2', 'seed': seed, 'generated_n': args.expected_n,
                  'min_length': args.min_length, 'reference_sha256': ref_hash,
                  'e1_manifest_sha256': sha256(manifest_file), 'methods': results}
        (out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summarize(all_rows, args.output_dir, selected_seeds, selected_methods)
    print(f'E2 COMPLETE: seeds={selected_seeds} methods={len(selected_methods)} output={args.output_dir}')


if __name__ == '__main__': main()
