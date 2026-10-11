#!/usr/bin/env python3
"""Descriptive figures from frozen evaluation CSVs; no model inference or selection.

Plot all seven models in fixed condition/seed order, never ordered by performance.
Use existing analysis validation helpers without changing its frozen protocol.
Clinical AD/CN endpoints remain separate and are not plotted by this script.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

import summarize_frozen_evaluation as a


BASELINE_SHA256 = 'a8cc03aea74f5310c1fc7936c72f94a44e55af11e2ada2a8ba5f8edd6d924db5'
SHORT = ['Mild', 'Moderate', 'Non', 'Very mild']
COLORS = ['#202020', '#0072B2', '#56B4E9', '#009E73', '#D55E00', '#CC79A7', '#E69F00']
NOTES = {
    'original100': 'Historically reused image benchmark: old model had 66 training, 9 validation and 25 test image matches. No independent clinical claim.',
    'internal_same_case': 'Full primary exact-row-held-out intersection is unchanged. Some similarity groups contain old development images; patient independence is unknown.',
    'old_hash_group_unexposed_sensitivity': 'Prespecified secondary subset of the full primary; excludes observed old development mates in prepared hash groups. Groups are not patients.',
}
TITLES = {'original100': 'Original 100: reused image benchmark',
          'internal_same_case': 'Full primary: unchanged same-case intersection',
          'old_hash_group_unexposed_sensitivity': 'Secondary sensitivity: no observed old development hash-group mate'}


def collect(args):
    """All case, label, probability and identity checks occur before any figure."""
    used = {}
    def use(path):
        path = Path(path).resolve()
        used[str(path)] = a.digest(path)
        return path
    ev, prepared = args.evaluation_dir, args.prepared
    meta = json.loads(use(ev / 'evaluation.json').read_text())
    selection = json.loads(use(args.validation_selection).read_text())
    if selection.get('frozen_before_heldout_evaluation') is not True:
        raise ValueError('Require selection frozen before held-out evaluation')
    if selection.get('analysis_protocol') != a.PROTOCOL or selection.get('analysis_script_sha256') != a.digest(a.__file__):
        raise ValueError('Frozen statistical protocol/code mismatch')
    checkpoints, frozen = meta.get('checkpoints', {}), selection.get('candidates', {})
    if len(frozen) != 6 or set(checkpoints) != {'old_baseline', *frozen}:
        raise ValueError('Require old baseline and exactly all six frozen candidates')
    if not re.fullmatch(r'[0-9a-f]{64}', args.baseline_checkpoint_sha256):
        raise ValueError('Invalid expected baseline checkpoint hash')
    if checkpoints['old_baseline'].get('sha256') != args.baseline_checkpoint_sha256:
        raise ValueError('Baseline checkpoint identity differs from pinned expected checkpoint')
    pairs = {}
    for name, record in frozen.items():
        if not re.fullmatch(r'candidate_[1-6]', name):
            raise ValueError('Unexpected frozen candidate identifier')
        info, pair = checkpoints[name], (record.get('mode'), record.get('seed'))
        cfg = info.get('selection_config', {})
        wanted = {(m, s) for m in a.PROTOCOL['planned_modes'] for s in a.PROTOCOL['planned_seeds']}
        if pair not in wanted or pair in pairs:
            raise ValueError('Unexpected/duplicate condition and seed')
        if (cfg.get('mode'), cfg.get('seed')) != pair or info.get('sha256') != record.get('checkpoint_sha256'):
            raise ValueError('Frozen/evaluated checkpoint identity mismatch')
        if not re.fullmatch(r'[0-9a-f]{64}', str(info.get('sha256', ''))):
            raise ValueError('Invalid candidate checkpoint hash')
        if cfg.get('smoke') is not False or cfg.get('test_used_for_selection') is not False:
            raise ValueError('Smoke or non-validation-selected candidate')
        pairs[pair] = name
    names = ['old_baseline'] + [pairs[(m, s)] for m in a.PROTOCOL['planned_modes'] for s in a.PROTOCOL['planned_seeds']]
    display = {'old_baseline': 'Old baseline'}
    for (mode, seed), name in pairs.items():
        display[name] = ('Stage only' if mode == 'stage_only' else 'Multitask') + f' | seed {seed}'

    stage_path, intersection_path, audit_path = [use(prepared / f) for f in ('stage_manifest.csv', 'same_case_test_intersection.csv', 'split_audit.json')]
    columns = ['path', 'label', 'group_id', 'split', 'baseline_strict_split']
    stage = a.index_rows(a.read_csv(stage_path, columns), 'path', 'stage manifest')
    intersection = a.index_rows(a.read_csv(intersection_path, columns), 'path', 'primary intersection')
    expected_primary = {p: r for p, r in stage.items() if r['split'] == 'test' and r['baseline_strict_split'] == 'test'}
    a.require_same(intersection, expected_primary, 'unchanged primary intersection')
    group_metadata = {}
    for r in stage.values():
        if not r['group_id'] or r['label'] not in {'0', '1', '2', '3'}:
            raise ValueError('Invalid stage group or label')
        identity = (r['split'], r['label'])
        if r['group_id'] in group_metadata and group_metadata[r['group_id']] != identity:
            raise ValueError('Stage group crosses splits or labels')
        group_metadata[r['group_id']] = identity
    for p, r in intersection.items():
        if any(r[k] != stage[p][k] for k in columns):
            raise ValueError('Intersection metadata mismatch')
    audit = json.loads(audit_path.read_text())
    if audit.get('manifest_sha256') != a.digest(stage_path) or audit.get('same_case_test_intersection_sha256') != a.digest(intersection_path):
        raise ValueError('Prepared manifest digest mismatch')
    for name in frozen:
        if checkpoints[name]['selection_config'].get('stage_manifest_sha256') != a.digest(stage_path):
            raise ValueError('Candidate stage manifest differs from prepared manifest')
    if not args.synthetic_qa and len(intersection) != 1058:
        raise ValueError('Production full primary must retain all 1058 rows')
    key = a.index_rows(a.read_csv(use(args.original100_key), ['FileName', 'TrueCategory']), 'FileName', 'original100 key')
    if len(key) != 100 or any(r['TrueCategory'] not in a.STAGES for r in key.values()):
        raise ValueError('Require exact 100-case key and known class labels')
    expectations = {'original100': {p: r['TrueCategory'] for p, r in key.items()},
                    'internal_same_case': {p: a.STAGES[int(r['label'])] for p, r in intersection.items()}}
    datasets = {}
    for cohort, expected in expectations.items():
        expected_files = {f'{name}_{cohort}_predictions.csv' for name in names}
        a.require_same({p.name for p in ev.glob(f'*_{cohort}_predictions.csv')}, expected_files, cohort + ' model files')
        ids = sorted(expected)
        data = {'ids': ids, 'y': np.array([a.STAGES.index(expected[p]) for p in ids]), 'models': {}}
        for name in names:
            path = use(ev / f'{name}_{cohort}_predictions.csv')
            index_key = 'FileName' if cohort == 'original100' else 'path'
            predicted = np.asarray(a.load_stage_predictions(path, expected, index_key))
            indexed = a.index_rows(a.read_csv(path, [index_key]), index_key, path.name)
            probabilities = np.asarray([a.validate_probabilities(indexed[p], ['P_' + cl for cl in a.STAGES], path.name) for p in ids])
            data['models'][name] = {'pred': predicted, 'prob': probabilities}
        datasets[cohort] = data
    sensitivity, pinned = args.old_hash_unexposed_manifest, selection.get('old_hash_unexposed_sensitivity')
    if bool(sensitivity) != bool(args.baseline_exposure_audit):
        raise ValueError('Sensitivity manifest and audit must be supplied together')
    if pinned and not sensitivity:
        raise ValueError('Frozen sensitivity subset must remain visible; supply its manifest and audit')
    if sensitivity:
        sp, ap = use(sensitivity), use(args.baseline_exposure_audit)
        if not pinned or pinned.get('manifest_sha256') != a.digest(sp) or pinned.get('audit_sha256') != a.digest(ap):
            raise ValueError('Sensitivity files differ from pre-evaluation frozen hashes')
        subset, _ = a.validate_sensitivity_manifest(sp, ap, stage_path, intersection_path, audit_path, stage, intersection)
        if not args.synthetic_qa and len(subset) != 756:
            raise ValueError('Production prespecified sensitivity must retain exactly 756 rows')
        primary = datasets['internal_same_case']
        positions = {p: i for i, p in enumerate(primary['ids'])}
        ids = sorted(subset); idx = np.array([positions[p] for p in ids])
        datasets['old_hash_group_unexposed_sensitivity'] = {
            'ids': ids, 'y': primary['y'][idx],
            'models': {name: {k: v[idx] for k, v in data.items()} for name, data in primary['models'].items()}}

    aligned_path = use(ev / 'original100_case_aligned_old_candidates_readers.csv')
    aligned = a.index_rows(a.read_csv(aligned_path, ['FileName', 'TrueCategory', *names]), 'FileName', 'aligned original100')
    a.require_same(aligned, key, 'aligned original100 cases')
    reader_names = sorted(k for k in next(iter(aligned.values())) if k.startswith('Reader_'))
    expected_readers = {'Reader_' + k for k in meta.get('reader_provenance', {})}
    a.require_same(reader_names, expected_readers, 'reader identities')
    if not reader_names:
        raise ValueError('No reader predictions found')
    reader_preds = {name: [] for name in reader_names}
    original = datasets['original100']
    for i, case in enumerate(original['ids']):
        row = aligned[case]
        if row['TrueCategory'] != key[case]['TrueCategory']:
            raise ValueError('Aligned true label mismatch')
        for name in names:
            if row[name] != a.STAGES[original['models'][name]['pred'][i]]:
                raise ValueError('Aligned model prediction differs from probability CSV')
        for name in reader_names:
            if row[name] not in a.STAGES:
                raise ValueError('Reader confusion figures require complete known labels on all 100 cases')
            reader_preds[name].append(a.STAGES.index(row[name]))
    return {'names': names, 'display': display, 'checkpoints': checkpoints, 'selection': selection,
            'datasets': datasets, 'readers': reader_preds, 'inputs_sha256': used}


def calibration(y, prob):
    """Multiclass sum Brier (0..2), top-label ECE with ten fixed equal-width bins."""
    onehot = np.eye(prob.shape[1])[y]
    brier = float(np.mean(np.sum((prob - onehot) ** 2, axis=1)))
    pred, confidence = prob.argmax(axis=1), prob.max(axis=1)
    bins = np.minimum((confidence * 10).astype(int), 9)
    records, ece = [], 0.
    for b in range(10):
        mask = bins == b; n = int(mask.sum())
        acc, conf = (float(np.mean(pred[mask] == y[mask])), float(confidence[mask].mean())) if n else (None, None)
        contribution = n / len(y) * abs(acc - conf) if n else 0.
        ece += contribution
        records.append({'lower': b / 10, 'upper': (b + 1) / 10, 'upper_inclusive': b == 9,
                        'n': n, 'accuracy': acc, 'mean_confidence': conf, 'ece_contribution': contribution})
    return {'multiclass_brier_sum': brier, 'top_label_ece_10_equal_width_bins': ece, 'bins': records}


def save_figure(fig, output, stem):
    paths = []
    for ext in ('png', 'pdf'):
        path = output / f'{stem}.{ext}'
        fig.savefig(path, dpi=300, facecolor='white')
        paths.append(path)
    return paths


def setup_plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    return plt


def confusion_panels(y, predictions, display, title, note, output, stem, synthetic):
    plt = setup_plotting()
    n = len(predictions); columns = min(4, n); rows = (n + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(max(10.5, 3.35 * columns), 3.65 * rows + 1.35), squeeze=False)
    matrices = {}
    for panel, (ax, (name, pred)) in enumerate(zip(axes.flat, predictions.items())):
        cm = confusion_matrix(y, pred, labels=range(4)); total = cm.sum(axis=1, keepdims=True)
        fraction = np.divide(cm, total, out=np.zeros_like(cm, dtype=float), where=total > 0)
        im = ax.imshow(fraction, vmin=0, vmax=1, cmap='Blues')
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if fraction[i, j] > .55 else '#202020', fontsize=11)
        ax.set_xticks(range(4), SHORT, rotation=32, ha='right')
        ax.set_yticks(range(4), SHORT)
        ax.set_xlabel('Predicted class')
        if panel % columns == 0:
            ax.set_ylabel('Reference class')
        ax.set_title(display[name].replace(' | ', '\n'), fontsize=10, pad=7)
        matrices[name] = cm.tolist()
    for ax in list(axes.flat)[n:]:
        ax.axis('off')
    header = ('SYNTHETIC QA — ' if synthetic else '') + title + f' | n={len(y)}'
    fig.suptitle(header, y=.986, fontsize=13, fontweight='bold')
    fig.text(.04, .91 if rows == 1 else .937, 'Cells show counts; color shows recall within each reference-class row. All models/readers retain their fixed order.', fontsize=9)
    fig.text(.04, .025, note, fontsize=8.7, va='bottom', wrap=True)
    fig.subplots_adjust(left=.095, right=.91, bottom=.23 if rows == 1 else .14, top=.78 if rows == 1 else .86, wspace=.48, hspace=.50)
    colorbar = fig.add_axes([.936, .30, .012, .40]); fig.colorbar(im, cax=colorbar, label='Reference-row fraction')
    paths = save_figure(fig, output, stem); plt.close(fig)
    return paths, matrices


def probability_panels(data, names, display, cohort, output, kind, synthetic):
    plt = setup_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.2), squeeze=False)
    title = TITLES[cohort] + (' | ROC' if kind == 'roc' else ' | precision–recall')
    fig.suptitle(('SYNTHETIC QA — ' if synthetic else '') + title + f' | n={len(data["y"])}', y=.99, fontsize=13, fontweight='bold')
    scores, handles = {}, []
    for k, ax in enumerate(axes.flat):
        binary = (data['y'] == k).astype(int); pos, neg = int(binary.sum()), int((binary == 0).sum())
        ax.set_title(f'{a.STAGES[k]} versus rest\npositive n={pos}; negative n={neg}', fontsize=11)
        scores[a.STAGES[k]] = {}
        for j, name in enumerate(names):
            prob = data['models'][name]['prob'][:, k]
            if not pos or not neg:
                scores[a.STAGES[k]][name] = None
                continue
            if kind == 'roc':
                xx, yy, _ = roc_curve(binary, prob, drop_intermediate=False)
                score = float(roc_auc_score(binary, prob))
            else:
                yy, xx, _ = precision_recall_curve(binary, prob)
                score = float(average_precision_score(binary, prob))
            draw = ax.plot if kind == 'roc' else ax.step
            extra = {} if kind == 'roc' else {'where': 'post'}
            line, = draw(xx, yy, color=COLORS[j], lw=1.5, ls='-' if j < 4 else '--', alpha=.9, **extra)
            if k == 0:
                handles.append(line)
            scores[a.STAGES[k]][name] = score
        if not pos or not neg:
            ax.text(.5, .5, 'Undefined: both target classes required', ha='center', va='center', transform=ax.transAxes)
        if kind == 'roc':
            ax.plot([0, 1], [0, 1], color='#BBBBBB', lw=.8, ls=':')
            ax.set_xlabel('False-positive rate'); ax.set_ylabel('True-positive rate')
        else:
            prevalence = pos / len(binary)
            ax.axhline(prevalence, color='#AAAAAA', lw=.8, ls=':')
            ax.text(.98, prevalence + .025, f'Prevalence {prevalence:.3f}', ha='right', va='bottom', fontsize=8, color='#666666')
            ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_xlim(-.015, 1.015); ax.set_ylim(-.015, 1.035)
        ax.grid(alpha=.16, lw=.6); ax.set_axisbelow(True)
    if handles:
        fig.legend(handles, [display[n] for n in names], loc='upper center', bbox_to_anchor=(.5, .953), ncol=3, fontsize=9, frameon=False)
    note = NOTES[cohort] + '\nClasswise one-versus-rest curves use frozen probabilities; no thresholds or models are chosen here. AUC/AP are descriptive, without new confidence intervals.'
    fig.text(.07, .025, note, fontsize=8.5, linespacing=1.5, va='bottom')
    fig.subplots_adjust(left=.07, right=.985, bottom=.12, top=.81, hspace=.36, wspace=.22)
    paths = save_figure(fig, output, f'{cohort}_{kind}_all_seven_models'); plt.close(fig)
    return paths, scores


def render(args):
    data = collect(args)
    if args.output.exists():
        raise ValueError('Choose a new figure output directory; previous figures are preserved')
    args.output.mkdir(parents=True)
    paths, metrics, calibration_rows, curve_rows = [], {}, [], []
    for cohort, dataset in data['datasets'].items():
        predictions = {n: dataset['models'][n]['pred'] for n in data['names']}
        files, cm = confusion_panels(dataset['y'], predictions, data['display'], TITLES[cohort], NOTES[cohort], args.output, f'{cohort}_confusion_all_seven_models', args.synthetic_qa)
        paths += files; metrics[cohort] = {'n': len(dataset['y']), 'confusion_counts': cm}
        for kind in ('roc', 'pr'):
            files, scores = probability_panels(dataset, data['names'], data['display'], cohort, args.output, kind, args.synthetic_qa)
            paths += files; metrics[cohort]['classwise_' + ('roc_auc' if kind == 'roc' else 'average_precision')] = scores
            for cl, model_scores in scores.items():
                for name, score in model_scores.items():
                    curve_rows.append({'cohort': cohort, 'n': len(dataset['y']), 'target_class': cl, 'checkpoint': name,
                                       'display_name': data['display'][name], 'metric': 'ROC_AUC' if kind == 'roc' else 'average_precision', 'value': score})
        metrics[cohort]['exploratory_descriptive_calibration'] = {}
        for name in data['names']:
            cal = calibration(dataset['y'], dataset['models'][name]['prob'])
            metrics[cohort]['exploratory_descriptive_calibration'][name] = cal
            calibration_rows.append({'cohort': cohort, 'n': len(dataset['y']), 'checkpoint': name, 'display_name': data['display'][name],
                                     'multiclass_brier_sum_0_to_2': cal['multiclass_brier_sum'],
                                     'top_label_ECE_10_equal_width_bins': cal['top_label_ece_10_equal_width_bins']})
    reader_names = {n: n.replace('Reader_', 'Reader ') for n in data['readers']}
    files, reader_cm = confusion_panels(data['datasets']['original100']['y'], data['readers'], reader_names,
        'Original 100: fixed reader panel', NOTES['original100'], args.output, 'original100_reader_confusion', args.synthetic_qa)
    paths += files
    with (args.output / 'exploratory_descriptive_calibration.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(calibration_rows[0])); writer.writeheader(); writer.writerows(calibration_rows)
    with (args.output / 'classwise_roc_auc_average_precision.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0])); writer.writeheader(); writer.writerows(curve_rows)
    report = {'status': 'synthetic_qa_not_research' if args.synthetic_qa else 'descriptive_figures_all_frozen_models',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': a.digest(__file__),
              'guard_helper_sha256': a.digest(a.__file__), 'numpy': np.__version__, 'sklearn': sklearn.__version__,
              'inputs_sha256': data['inputs_sha256'], 'checkpoint_provenance': data['checkpoints'],
              'fixed_plot_order': data['names'], 'display_names': data['display'],
              'new_model_inference': False, 'test_based_selection': False, 'threshold_optimization': False,
              'statistical_protocol_unchanged': True,
              'calibration_protocol': {'status': 'additional exploratory descriptive visualization summary, separate from frozen statistical protocol',
                  'brier': 'Mean over images of SUM over 4 classes (probability minus one-hot target)^2; range 0 to 2; not divided by class count.',
                  'ece': 'Top-label expected calibration error: sum over 10 fixed equal-width confidence bins of (bin count / n) * abs(bin accuracy - bin mean maximum probability).',
                  'bins': '[0,.1), [.1,.2), ..., [.9,1]; empty bins contribute zero; confidence 1 goes in final bin.',
                  'limits': 'No new intervals, significance tests or calibration fitting. ECE is bin/sample dependent and can mask classwise miscalibration. Neither metric establishes clinical reliability.'},
              'clinical_endpoint': 'Not plotted; AD/CN cohorts remain separate in frozen statistical analysis. No pooling with four-stage image outcomes.',
              'datasets': metrics, 'reader_original100_confusion_counts': reader_cm,
              'figures': {p.name: {'sha256': a.digest(p), 'bytes': p.stat().st_size} for p in paths}}
    (args.output / 'evaluation_figure_provenance.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-dir', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--original100-key', type=Path, required=True)
    p.add_argument('--validation-selection', type=Path, required=True)
    p.add_argument('--old-hash-unexposed-manifest', type=Path)
    p.add_argument('--baseline-exposure-audit', type=Path)
    p.add_argument('--baseline-checkpoint-sha256', default=BASELINE_SHA256)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--synthetic-qa', action='store_true', help='Mark all figures as synthetic and relax only production cohort counts for artificial QA fixtures')
    args = p.parse_args()
    result = render(args)
    print(json.dumps({'status': result['status'], 'figures': len(result['figures']), 'output': str(args.output.resolve())}))


if __name__ == '__main__':
    main()
