"""Reconstruct specified evaluation dates from the next complete daily sample batch."""
import argparse
import csv
import hashlib
import math
from datetime import date
from pathlib import Path
from statistics import mean, median

from review_contextualization import read, write


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combine(paths):
    samples = []
    for path in paths:
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        days = [r['date'] for r in rows]
        if days != sorted(set(days)):
            raise ValueError(f'Unordered or duplicate dates: {path}')
        values = {r['date']: float(r['value']) for r in rows}
        if any(not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError(f'Invalid values: {path}')
        samples.append(values)
    common = set.intersection(*(set(s) for s in samples))
    return {d: mean(s[d] for s in samples) if all(s[d] > 0 for s in samples) else 0.0
            for d in sorted(common)}


def reconstruct(original, sample, target, excluded, minimum_overlap):
    """Align scales by date using positive preceding overlap, as in daily stitching."""
    ratios = [original[d] / v for d, v in sample.items()
              if d < target and d not in excluded and v > 0 and original.get(d, 0) > 0]
    if sample[target] == 0:
        return 0.0, None, len(ratios)
    if len(ratios) < minimum_overlap:
        return None, None, len(ratios)
    scale = median(ratios)
    return sample[target] * scale, scale, len(ratios)


def next_samples(raw, country, target, count):
    for batch in sorted(raw.glob('output_*')):
        collected = batch.name.removeprefix('output_')
        if collected <= target:
            continue
        # Only the first complete post-date batch is needed, not settled weeks-later data.
        for first in sorted((batch/'sample0'/country).glob('*_multiTimeline.csv')):
            paths = [batch/f'sample{i}'/country/first.name for i in range(count)]
            if not all(p.is_file() for p in paths):
                continue
            values = combine(paths)
            if target in values:
                return batch, first.name, paths, values
    raise ValueError(f'No complete later sample for {country}/{target}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--raw-directory', type=Path, required=True)
    parser.add_argument('--dates', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-count', type=int, default=45)
    parser.add_argument('--minimum-overlap', type=int, default=10)
    args = parser.parse_args()
    if args.sample_count < 1 or args.minimum_overlap < 1:
        parser.error('Sample count and minimum overlap must be positive')
    root, output = args.results.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        parser.error('Prepare repairs outside the results directory')
    config = read(root/'experiment.json')
    if len(config['seeds']) != 1:
        parser.error('Evaluation reconstruction expects a single-seed results directory')
    targets = sorted(set(args.dates))
    if any(not config['train_end'] < d <= config['test_end'] for d in targets):
        parser.error('Repair dates must be in the evaluation period')
    for d in targets:
        date.fromisoformat(d)
    (output/'series').mkdir(parents=True, exist_ok=True)
    records, hashes = [], {}
    for country in config['countries']:
        source = root/f'inputs/{country}.json'
        rows = read(source)
        original = {r['date']: r['rate'] for r in rows}
        hashes[country] = fingerprint(source)
        replacements = {}
        for target in targets:
            batch, window, paths, sample = next_samples(args.raw_directory, country, target, args.sample_count)
            value, scale, anchors = reconstruct(original, sample, target, targets, args.minimum_overlap)
            records.append(dict(country=country, date=target, original=original[target], replacement=value,
                batch=batch.name, window=window, samples=len(paths), raw_value=sample[target],
                scale=scale, anchor_pairs=anchors, status='reconstructed' if value is not None else 'insufficient_overlap',
                sample_sha256=[fingerprint(p) for p in paths]))
            if value is not None:
                replacements[target] = value
        candidate = [dict(r, rate=replacements[r['date']]) if r['date'] in replacements else r for r in rows]
        assert all(a == b for a, b in zip(rows, candidate) if a['date'] not in targets)
        write(output/'series'/f'{country}.json', candidate)
        print(country, 'prepared', flush=True)
    write(output/'data-quality.json', dict(method='Next complete post-date batch; any-zero aggregation; '
        'median positive preceding overlap ratio, excluding repair dates', dates=targets,
        sample_count=args.sample_count, minimum_overlap=args.minimum_overlap,
        source_sha256=hashes, training_unchanged=True, records=records))


if __name__ == '__main__':
    main()
