"""Evaluate saved Pareto parameters for coverage and failed-run diagnostics."""
import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from statistics import median

from batch_ensemble_tuning import label_applies, required_labels
from review_contextualization import (
    aggregate_rows, coverage, filter_output, matches, read, retain_by_impact, write,
)

VARIANTS = ['original', 'labels', 'ensemble', 'labels_ensemble']
FRACTIONS = [0, .05, .1, .25, .5, .75, 1]


def validate_cache(directory, identity):
    manifest = directory / 'manifest.json'
    if manifest.exists():
        if read(manifest) != identity:
            raise ValueError('Review inputs changed; use a new output directory')
    elif any(directory.iterdir()):
        raise ValueError('Unversioned review cache; use a new output directory')
    else:
        write(manifest, identity)


def parameters(values, variant):
    names = (['window', 'min_residual_ratio', 'efficiency', 'chebyshev_z',
              'chebyshev_k', 'isolation_forest_min_score', 'lof_min_score']
             if 'ensemble' in variant else
             ['window', 'z', 'k', 'min_residual_ratio', 'efficiency'])
    return dict(zip(names, [round(values[0]), *values[1:]]))


def unique_front(front, variant):
    result, seen = [], set()
    for index, (values, objectives) in enumerate(front):
        params = parameters(values, variant)
        key = json.dumps(params, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(dict(front_index=index, parameters=params,
                               training_count=objectives[0], training_impact=-objectives[1]))
    return sorted(result, key=lambda p: (p['training_count'], -p['training_impact'], p['front_index']))


def sample_front(front, fraction):
    return front[math.floor(fraction * (len(front) - 1) + .5)]


def export_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


class Review:
    def __init__(self, results, output, binary, minimum_countries=4):
        self.root, self.output, self.binary = results, output, binary
        self.minimum_countries = minimum_countries
        self.config = read(results / 'experiment.json')
        self.seed, = self.config['seeds']
        self.cutoff, self.end = self.config['train_end'], self.config['test_end']
        self.proximity = self.config['proximity_days']
        self.labels = read(results / 'inputs/labels.json')
        self.training = read(results / 'inputs/training-labels.json')
        self.runs = {(c, v): read(self.directory(c, v) / 'result.json')
                     for c in self.config['countries'] for v in VARIANTS}
        self.common = [c for c in self.config['countries'] if all(
            self.runs[c, v]['status'] == 'completed' for v in VARIANTS)]
        self.cache = output / 'cache'
        self.cache.mkdir(parents=True, exist_ok=True)
        sources = [results / 'experiment.json', results / 'inputs/labels.json',
                   results / 'inputs/training-labels.json', binary]
        sources += [results / f'inputs/{c}.json' for c in self.config['countries']]
        sources += [self.directory(c, v) / name for c, v in self.runs
                    for name in ('result.json', 'pareto-front.json')
                    if (self.directory(c, v) / name).exists()]
        sources += [Path(__file__).with_name(name) for name in
                    ('review_saved_fronts.py', 'review_contextualization.py', 'batch_ensemble_tuning.py')]
        validate_cache(self.cache, {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sources})

    def directory(self, country, variant):
        return self.root / f'runs/{self.seed}/{country}/{variant}'

    def front(self, country, variant):
        return unique_front(read(self.directory(country, variant) / 'pareto-front.json'), variant)

    def detect(self, country, variant, point, annotated=False):
        suffix = '-annotated' if annotated else ''
        cache = self.cache / country / variant
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"{point['front_index']}{suffix}.json"
        if path.exists():
            stored = read(path)
            if stored['parameters'].keys() != point['parameters'].keys() or any(
                not math.isclose(stored['parameters'][k], v, rel_tol=1e-14, abs_tol=1e-14)
                for k, v in point['parameters'].items()
            ):
                raise ValueError(f'Parameter cache mismatch: {path}')
            return stored
        with tempfile.TemporaryDirectory() as tmp:
            params = Path(tmp) / 'parameters.json'
            write(params, point['parameters'])
            command = [str(self.binary), 'detect', '--algorithm',
                       'ensemble' if 'ensemble' in variant else 'chebyshev',
                       '--input', str(self.root / f'inputs/{country}.json'),
                       '--parameters', str(params), '--reference-end', self.cutoff]
            if annotated:
                command.append('--include-annotated')
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            data = json.loads(result.stdout)
            write(path, data)
            return data

    def required(self, country, training):
        run = self.runs[country, 'original']
        return required_labels(self.training if training else self.labels, country,
                               run['first_date'] if training else run['test_first_date'],
                               self.cutoff if training else self.end, 0)

    def evaluate_point(self, task):
        country, variant, point = task
        cache = self.cache / country / variant / f"{point['front_index']}-context.json"
        if cache.exists():
            return read(cache)
        data = self.detect(country, variant, point)
        train = [a for a in data['anomalies'] if a['end'] <= self.cutoff]
        raw = [a for a in data['anomalies'] if self.cutoff < a['start'] <= self.end]
        train_starts = [a for a in data['anomalies'] if a['start'] <= self.cutoff]
        assert len(train_starts) == int(point['training_count']), (country, variant, point['front_index'], len(train_starts), point['training_count'])
        if all(a['end'] <= self.cutoff for a in train_starts):
            assert math.isclose(sum(a['impact'] for a in train_starts), point['training_impact'],
                                abs_tol=1e-7, rel_tol=1e-7), (country, variant, point['front_index'], 'impact')
        with tempfile.TemporaryDirectory() as tmp:
            calibrated, threshold = filter_output(self.binary, country, train,
                self.root / 'inputs/training-labels.json', self.proximity, 1, Path(tmp))
            coalesced, _ = filter_output(self.binary, country, raw,
                self.root / 'inputs/training-labels.json', self.proximity, 1, Path(tmp))
        high = median(a['impact'] for a in calibrated) if calibrated else None
        retained = retain_by_impact(coalesced, threshold, high)
        train_labels, test_labels = self.required(country, True), self.required(country, False)
        row = dict(country=country, variant=variant, selection='front_sample',
            front_index=point['front_index'], parameters=point['parameters'],
            training_count=len(train_starts), training_impact=point['training_impact'],
            training_labels_total=len(train_labels),
            training_labels_covered=len(coverage(train_starts, train_labels, self.proximity)),
            labels_total=len(test_labels), raw=len(raw),
            raw_sparse=sum(a.get('sparse', False) for a in raw), coalesced=len(coalesced),
            retained=len(retained), raw_covered=len(coverage(raw, test_labels, self.proximity)),
            retained_covered=len(coverage(retained, test_labels, self.proximity)),
            threshold=threshold, high_impact_threshold=high, retained_anomalies=retained)
        if variant in ('labels', 'labels_ensemble'):
            assert row['training_labels_total'] == row['training_labels_covered']
        write(cache, row)
        return row

    def curves(self, workers):
        if not self.common:
            raise ValueError('No countries completed all four configurations')
        fronts = {(c, v): self.front(c, v) for c in self.common for v in VARIANTS}
        selected = {(c, v, f): sample_front(front, f) for (c, v), front in fronts.items()
                    for f in FRACTIONS}
        tasks = {(c, v, p['front_index']): (c, v, p) for (c, v, _), p in selected.items()}
        done = {}
        with ThreadPoolExecutor(workers) as pool:
            for n, row in enumerate(pool.map(self.evaluate_point, tasks.values()), 1):
                done[row['country'], row['variant'], row['front_index']] = row
                if n % 100 == 0:
                    print(f'Pareto review: {n}/{len(tasks)} saved parameter sets', flush=True)
        labels_by_country = {c: [l for l in self.labels if self.cutoff < l['date'] <= self.end
                                 and label_applies(l, c)] for c in self.common}
        totals, detail = [], []
        for variant in VARIANTS:
            for f in FRACTIONS:
                rows = [done[c, variant, selected[c, variant, f]['front_index']] for c in self.common]
                with tempfile.TemporaryDirectory() as tmp:
                    summary, _ = aggregate_rows(rows, self.binary, labels_by_country,
                                                self.proximity, Path(tmp), self.minimum_countries)
                totals.append(dict(variant=variant, front_fraction=f, countries=len(rows),
                    training_anomalies=sum(r['training_count'] for r in rows),
                    test_anomalies=summary['raw'], test_labels=summary['labels_total'],
                    raw_covered=summary['raw_covered'],
                    raw_coverage=summary['raw_covered']/summary['labels_total'] if summary['labels_total'] else None,
                    minimum_group_countries=self.minimum_countries,
                    notifications=summary['notifications'][str(self.minimum_countries)],
                    retained_covered=summary['retained_covered'],
                    retained_coverage=summary['retained_covered']/summary['labels_total'] if summary['labels_total'] else None))
                for row in rows:
                    detail.append(dict(front_fraction=f, **{k:v for k,v in row.items()
                        if k not in ('parameters', 'retained_anomalies')}))
        export_csv(self.output / 'coverage-curves.csv', totals)
        export_csv(self.output / 'sampled-points.csv', detail)
        write(self.output / 'curve-data.json', totals)
        print('Coverage curves complete', flush=True)

    def diagnostics(self, workers):
        failed = [(c, v) for (c, v), run in self.runs.items() if run['status']=='no_feasible_solution']
        summaries, labels_table, missed, parameters_table, scores, windows = [], [], [], [], [], []
        for country, failed_variant in sorted(failed):
            source = 'ensemble' if 'ensemble' in failed_variant else 'original'
            front = self.front(country, source)
            required = self.required(country, True)
            def evaluate(point):
                data = self.detect(country, source, point)
                anomalies = [a for a in data['anomalies'] if a['start'] <= self.cutoff]
                covered = [i for i,l in enumerate(required) if any(matches(a,l,self.proximity) for a in anomalies)]
                return dict(point=point, covered=covered)
            with ThreadPoolExecutor(workers) as pool:
                candidates = list(pool.map(evaluate, front))
            best = min(candidates, key=lambda c: (-len(c['covered']), c['point']['training_count'],
                                                  -c['point']['training_impact'], c['point']['front_index']))
            best_count = len(best['covered'])
            tied = [c for c in candidates if len(c['covered']) == best_count]
            summaries.append(dict(country=country, failed_configuration=failed_variant,
                diagnostic_source=source, saved_candidates=len(candidates), required_labels=len(required),
                best_covered=best_count, best_candidates=len(tied),
                representative_front_index=best['point']['front_index']))
            params = best['point']['parameters']
            best_anomalies = [a for a in self.detect(country, source, best['point'])['anomalies']
                              if a['start'] <= self.cutoff]
            parameters_table.append(dict(country=country, failed_configuration=failed_variant,
                diagnostic_source=source, front_index=best['point']['front_index'], **params))
            missed_here = []
            for i,l in enumerate(required):
                closest = min(best_anomalies, key=lambda a: abs((date.fromisoformat(a['start'])-
                    date.fromisoformat(l['date'])).days), default=None)
                row = dict(country=country, failed_configuration=failed_variant,
                    label_date=l['date'], event=l.get('title') or l.get('description',''),
                    candidate_count=len(candidates), candidates_covering=sum(i in c['covered'] for c in candidates),
                    best_candidates=len(tied), best_candidates_covering=sum(i in c['covered'] for c in tied),
                    representative_covers=i in best['covered'], representative_front_index=best['point']['front_index'],
                    nearest_start=closest['start'] if closest else None,
                    nearest_distance_days=(date.fromisoformat(closest['start'])-date.fromisoformat(l['date'])).days if closest else None,
                    inside_anomaly=any(a['start']<=l['date']<=a['end'] for a in best_anomalies))
                labels_table.append(row)
                if i not in best['covered']:
                    missed.append(row)
                    missed_here.append(l)
            data = self.detect(country, source, best['point'], annotated=True)
            # Evaluate the other members with the same shared parameters. Their state
            # evolves independently; ensemble-state votes are also exported separately.
            ensemble_reference = (params if source=='ensemble' else
                read(self.directory(country,'ensemble') / 'knee-parameters.json'))
            standalone = {}
            for algorithm in ['chebyshev', 'isolation-forest', 'lof']:
                p = {k:params[k] for k in ['window','min_residual_ratio','efficiency']}
                if algorithm=='chebyshev':
                    p.update(z=params.get('z',params.get('chebyshev_z')),
                             k=params.get('k',params.get('chebyshev_k')))
                else:
                    p['min_score'] = ensemble_reference['isolation_forest_min_score' if algorithm=='isolation-forest' else 'lof_min_score']
                with tempfile.TemporaryDirectory() as tmp:
                    file = Path(tmp) / 'parameters.json'; write(file,p)
                    result = subprocess.run([str(self.binary),'detect','--algorithm',algorithm,
                        '--input',str(self.root/f'inputs/{country}-train.json'),'--parameters',str(file),
                        '--include-annotated'],capture_output=True,text=True)
                if result.returncode:
                    raise RuntimeError(f'{country}/{algorithm}: {result.stderr.strip()}')
                standalone[algorithm] = json.loads(result.stdout)['annotated']
                parameters_table.append(dict(country=country,failed_configuration=failed_variant,
                    diagnostic_source='standalone_'+algorithm,front_index=best['point']['front_index'],**p))
            for j,a in enumerate(data['annotated']):
                if a['date'] > self.cutoff:
                    break
                row = dict(country=country, failed_configuration=failed_variant,
                    date=a['date'], value=a['value'], diagnostic_anomaly=a['anomaly'])
                for alg, observations in standalone.items():
                    r=observations[j];assert r['date']==a['date']
                    sparse=r['demand_category'] in ('none','lumpy','intermittent')
                    evaluated=r['value']>0 and not sparse and r['demand_category'] is not None
                    row.update({alg+'_score':r['score'] if evaluated else None,
                                alg+'_cutoff':r['minimum_score'] if evaluated else None,
                                alg+'_anomaly':r['anomaly'], alg+'_state':r['demand_category'],
                                alg+'_residual':r['residual'] if sparse else None})
                vote=a.get('ensemble_vote')
                for name in ['chebyshev','isolation_forest','lof']:
                    row['ensemble_'+name+'_score']=vote[name]['score'] if vote else None
                    row['ensemble_'+name+'_vote']=vote[name]['voted'] if vote else None
                scores.append(row)
                for l in missed_here:
                    delta=(date.fromisoformat(a['date'])-date.fromisoformat(l['date'])).days
                    if abs(delta)<=3:
                        windows.append(dict(label_date=l['date'],event=l.get('title') or l.get('description',''),
                                            offset_days=delta,**row))
            print(f"Diagnostics {country}/{failed_variant}: best {best_count}/{len(required)}",flush=True)
        for name, rows in [('failed-runs',summaries),('training-label-coverage',labels_table),
                           ('missed-labels',missed),('parameters',parameters_table),
                           ('daily-scores',scores),('missed-label-scores',windows)]:
            export_csv(self.output/f'{name}.csv',rows)
        return dict(failures=summaries,missed=missed,windows=windows,parameters=parameters_table,coverage=labels_table)

    def vote_cases(self):
        rows=[]
        for c in self.config['countries']:
            if self.runs[c,'labels_ensemble']['status']!='completed':continue
            data=read(self.directory(c,'labels_ensemble')/'knee-detection.json')
            for a in data['anomalies']:
                v=a.get('ensemble_vote')
                if not self.cutoff<a['start']<=self.end or not v:continue
                if v['chebyshev']['voted'] or not (v['isolation_forest']['voted'] and v['lof']['voted']):continue
                labels=[l for l in self.labels if self.cutoff<l['date']<=self.end
                        and label_applies(l,c) and matches(a,l,self.proximity)]
                row=dict(country=c,start=a['start'],end=a['end'],peak=a['peak'],impact=a['impact'],
                    matched=bool(labels),label_matches='; '.join(f"{l['date']} [{l['scope']}] {l.get('title') or l.get('description','')}" for l in labels))
                for name in ['chebyshev','isolation_forest','lof']:
                    row[name+'_score']=v[name]['score']; row[name+'_cutoff']=v[name]['minimum_score']
                rows.append(row)
        export_csv(self.output/'iforest-lof-cases.csv',rows)
        return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--minimum-countries', type=int, default=4,
                        help='Minimum countries for a same-day alert group (default: 4)')
    parser.add_argument('--stage',choices=['diagnostics','curves','all'],default='all')
    parser.add_argument('--binary',type=Path,default=Path(__file__).resolve().parents[1]/'target/release/cenalert-detector')
    args=parser.parse_args()
    if args.workers < 1 or args.minimum_countries < 2:
        parser.error('--workers must be positive and --minimum-countries must be at least 2')
    args.output.mkdir(parents=True,exist_ok=True)
    review=Review(args.results.resolve(),args.output.resolve(),args.binary.resolve(),args.minimum_countries)
    if args.stage in ('diagnostics','all'):
        result=review.diagnostics(args.workers);result['vote_cases']=review.vote_cases()
        write(args.output/'workbook-data.json',result)
        shutil.copyfile(args.results/'inputs/labels.json',args.output/'labels.json')
    if args.stage in ('curves','all'):review.curves(args.workers)


if __name__=='__main__':main()
