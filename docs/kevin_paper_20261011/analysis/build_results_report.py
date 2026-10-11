#!/usr/bin/env python3
"""Render verified Kevin experiment evidence; no inference, fitting or selection.

Standard library only. The uncertainty analysis must already be complete. This
program checks local checkpoints and source digests, then formats existing
results. It never loads checkpoint tensors or fabricates missing metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

MODES = ('stage_only', 'multitask')
SEEDS = (20261011, 20261012, 20261013)
METRICS = ('accuracy', 'balanced_accuracy', 'macro_f1')
STAGES = ('MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented')
COHORTS = ('internal_same_case', 'old_hash_group_unexposed_sensitivity', 'original100')
TITLES = {
    'internal_same_case': 'Primary exact-row-held-out legacy comparison',
    'old_hash_group_unexposed_sensitivity': 'Secondary old-hash-group-unexposed sensitivity analysis',
    'original100': 'Historical original100 comparison',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def esc(value):
    return str(value).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')


def link(path, label=None):
    path = Path(path).resolve()
    return '[' + esc(label or path.name) + '](<' + str(path).replace('>', '%3E') + '>)'


def number(value, digits=4):
    if value is None:
        return 'undefined'
    require(isinstance(value, (int, float)) and math.isfinite(value), 'Nonfinite/non-numeric result')
    return f'{value:.{digits}f}'


def pct(value):
    return 'undefined' if value is None else number(100 * value, 2) + '%'


def ci(value, percentage=False):
    bounds = value.get('ci95')
    if bounds is None:
        return 'not estimable'
    fmt = pct if percentage else number
    text = '[' + fmt(bounds[0]) + ', ' + fmt(bounds[1]) + ']'
    if value.get('degenerate'):
        text += ' (degenerate)'
    return text


def metric_cell(model, metric):
    value = model['point'][metric]
    return pct(value) + '; ' + ci(model['bootstrap'][metric], True)


def diff_cell(value):
    estimate = value['estimate']
    point = 'undefined' if estimate is None else f'{100 * estimate:+.2f} pp'
    bounds = value.get('ci95')
    interval = 'not estimable' if bounds is None else '[' + ', '.join(f'{100*x:+.2f}' for x in bounds) + '] pp'
    return point + '; ' + interval + (' (degenerate)' if value.get('degenerate') else '')


def exact_cell(value):
    return f"{value['successes']}/{value['n']} ({pct(value['estimate'])}); {ci(value, True)}"


def table(headers, rows):
    return '\n'.join([
        '| ' + ' | '.join(map(esc, headers)) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
        *('| ' + ' | '.join(map(esc, row)) + ' |' for row in rows),
    ])


def bound_input(analysis, path):
    """Allow retrieval relocation only for an unambiguous source basename."""
    path = Path(path).resolve()
    matches = [r for r in analysis['input_manifest'] if Path(r['path']).name == path.name]
    require(len(matches) == 1, 'Ambiguous/missing analysis input binding: ' + path.name)
    require(matches[0]['sha256'] == digest(path), 'Analysis input changed: ' + path.name)


def verify(args):
    root, ev = args.results_root.resolve(), args.evaluation_dir.resolve()
    analysis = read_json(args.uncertainty_json)
    meta = read_json(ev / 'evaluation.json')
    selection = read_json(args.validation_selection)
    suite = read_json(root / 'suite_status.json')
    require(analysis['protocol']['version'] == 2, 'Require frozen analysis protocol version 2')
    require(analysis['validation_only_preference'] == selection, 'Validation selection differs from analyzed record')
    require(analysis['checkpoint_provenance'] == meta['checkpoints'], 'Checkpoint provenance differs from analysis')
    require(selection.get('frozen_before_heldout_evaluation') is True, 'Missing frozen selection provenance')
    bound_input(analysis, ev / 'evaluation.json')
    bound_input(analysis, args.validation_selection)
    require(set(COHORTS) == set(analysis['stage']), 'Require primary, sensitivity and original100 results')
    require(analysis['stage']['original100']['n'] == 100, 'Original100 must contain exactly 100 images')
    names = list(selection['candidates'])
    require(len(names) == 6 and set(meta['checkpoints']) == set(names) | {'old_baseline'}, 'Require six candidates and old baseline')
    expected_runs = {f'{m}_seed{s}' for m in MODES for s in SEEDS}
    require(set(suite['completed']) == expected_runs and len(suite['completed']) == 6, 'Suite does not contain six completed runs')
    require(suite['state'] in {'training_complete_awaiting_frozen_evaluation', 'complete', 'evaluation_complete', 'retrieved_and_cleaned'}, 'Suite is unfinished or failed')
    runs, pairs = {}, set()
    for name in names:
        record = selection['candidates'][name]
        info = meta['checkpoints'][name]
        cfg = info['selection_config']
        pair = (cfg['mode'], cfg['seed'])
        require(pair not in pairs and pair[0] in MODES and pair[1] in SEEDS, 'Duplicate/unexpected condition or seed')
        pairs.add(pair)
        require(not cfg.get('smoke') and cfg.get('test_used_for_selection') is False, 'Not a research validation-selected run')
        require(record['mode'] == pair[0] and record['seed'] == pair[1], 'Selection run identity mismatch')
        require(record['checkpoint_sha256'] == info['sha256'], 'Selection checkpoint digest mismatch')
        run = root / f'{pair[0]}_seed{pair[1]}'
        require(digest(run / 'best.pt') == info['sha256'], 'Local best checkpoint digest mismatch: ' + name)
        config, history = read_json(run / 'config.json'), read_json(run / 'history.json')
        for key in ('mode', 'seed', 'stage_manifest_sha256', 'clinical_manifest_sha256'):
            require(config.get(key) == cfg.get(key), 'Run config identity mismatch: ' + key)
        require(bool(history) and [r['epoch'] for r in history] == list(range(1, len(history)+1)), 'Incomplete/duplicate epoch history')
        epoch = int(record['epoch'])
        require(epoch == info['epoch'] and 1 <= epoch <= len(history), 'Selected epoch mismatch')
        require(abs(history[epoch-1]['val_stage_loss'] - record['stage_validation_nll']) <= 1e-8, 'Selected validation NLL mismatch')
        for item in ('history.csv', 'loss_curves.png', 'accuracy_curves.png'):
            require((run / item).is_file(), 'Missing raw run artifact: ' + str(run / item))
        runs[name] = {'directory': run, 'config': config, 'history': history, 'selection': record}
    for cohort in COHORTS:
        require(set(analysis['stage'][cohort]['models']) == set(names) | {'old_baseline'}, 'Incomplete model results: ' + cohort)
    for path in ev.glob('*_predictions.csv'):
        bound_input(analysis, path)
    aligned_path = ev / 'original100_case_aligned_old_candidates_readers.csv'
    bound_input(analysis, aligned_path)
    with aligned_path.open(newline='', encoding='utf-8-sig') as f:
        aligned = list(csv.DictReader(f))
    require(len(aligned) == 100 and len({r['FileName'] for r in aligned}) == 100, 'Invalid case-aligned original100')
    readers = sorted(analysis['readers']['paired_comparisons'])
    require(set(k for k in aligned[0] if k.startswith('Reader_')) == set(readers), 'Reader columns mismatch')
    for name in ['old_baseline'] + names:
        correct = sum(r[name] == r['TrueCategory'] for r in aligned)
        require(correct == analysis['stage']['original100']['models'][name]['point']['correct'], 'Original100 correctness mismatch')
    return analysis, meta, selection, suite, runs, aligned, readers


def render(args):
    a, meta, selection, suite, runs, aligned, readers = verify(args)
    names = list(runs)
    labels = {'old_baseline': 'Old baseline'}
    labels.update({n: f"{n}: {r['selection']['mode']}, seed {r['selection']['seed']}" for n, r in runs.items()})
    lines = []
    def add(text=''):
        lines.append(text)
    def section(title):
        add('## ' + title); add()
    def add_table(headers, rows):
        add(table(headers, rows)); add()
    add('# Kevin dementia experiment results'); add()
    add('This report renders the frozen six-run experiment and its recorded uncertainty analysis. It performs no inference or model selection. All conditions, seeds and benchmark results are retained; numerical differences alone do not establish clinical superiority.'); add()
    methods = Path(__file__).with_name('methods_protocol_snapshot.md')
    if methods.is_file():
        add('Manuscript Methods draft: ' + link(methods) + '; SHA256 `' + digest(methods) + '`.'); add()
    section('Evidence and validation-only preference')
    add(f"Suite status: **{esc(suite['state'])}**. Six completed runs are recorded. The condition preferred before held-out evaluation was **{esc(selection['preferred_mode'])}**; its exploratory representative was **{esc(selection['exploratory_representative_candidate'])}**. This preference used mean stage-validation NLL and then the representative seed's validation NLL, never test outcomes."); add()
    add_table(['Evidence', 'SHA256'], [(link(p), digest(p)) for p in (args.results_root/'suite_status.json', args.validation_selection, args.evaluation_dir/'evaluation.json', args.uncertainty_json)])
    add('Checkpoint selection required an NLL improvement greater than the configured minimum delta. It need not equal the mathematical minimum observed NLL when later improvements were smaller.'); add()
    section('All six training runs and raw curves')
    rows = []
    for name, run in runs.items():
        cfg, record, folder = run['config'], run['selection'], run['directory']
        rows.append([name, record['mode'], record['seed'], len(run['history']), record['epoch'], number(record['stage_validation_nll'], 6), cfg['epochs'], cfg['min_epochs'], cfg['patience'], number(cfg['min_delta'], 6), link(folder/'loss_curves.png', 'loss'), link(folder/'accuracy_curves.png', 'accuracy'), link(folder/'history.csv', 'raw CSV')])
    add_table(['Candidate', 'Condition', 'Seed', 'Epochs completed', 'Selected epoch', 'Selected validation NLL', 'Maximum epochs', 'Minimum epochs', 'Patience', 'Minimum delta', 'Raw loss curves', 'Raw accuracy curves', 'History'], rows)
    add('Stage train loss logs unweighted, unsmoothed cross-entropy; optimization uses inverse-group/class weights and label smoothing. Augmentation and dropout are active during training. Clinical train loss uses balanced sampled slices, while validation averages nine slice probabilities per source subject before scoring. The plotted training/validation losses therefore have different sampling and aggregation conditions. Raw curves are linked without cosmetic smoothing.'); add()
    for cohort in COHORTS:
        s = a['stage'][cohort]
        section(TITLES[cohort])
        add(f"**{s['n']} images; {s['n_resampling_units']} resampling units.** {esc(s.get('interpretation', ''))}"); add()
        if cohort == 'internal_same_case':
            exp = a['baseline_group_exposure']
            add(f"Observed old-development similarity overlap: {exp['images_with_old_train_group_mate']} primary images have an old training group mate; {exp['images_with_old_train_or_validation_group_mate']} have an old training or validation group mate. These are coarse image-similarity groups, not patient IDs."); add()
        elif cohort == 'old_hash_group_unexposed_sensitivity':
            add('The secondary subset excludes observed old training/validation group mates and reuses existing primary predictions. The complete primary remains above. Conservative dHash collisions can remove unrelated people; this subset does not establish patient independence.'); add()
        else:
            add('These historical cases match 66 training, 9 validation and 25 test rows in the old strict manifest. This descriptive benchmark cannot establish independent clinical superiority over readers.'); add()
        add_table(['Model', 'Correct / n', 'Accuracy; 95% interval', 'Balanced accuracy; 95% interval', 'Macro F1; 95% interval'], [[labels[n], f"{s['models'][n]['point']['correct']}/{s['models'][n]['point']['n']}", *[metric_cell(s['models'][n], m) for m in METRICS]] for n in ['old_baseline']+names])
        add('Per-class recalls expose class-specific regressions; each cell shows correct class predictions / true class count.'); add()
        rows = []
        for name in ['old_baseline'] + names:
            point = s['models'][name]['point']
            counts = point['class_counts']
            rows.append([labels[name]] + [f"{point['confusion_matrix'][i][i]}/{counts[i]} ({pct(point['confusion_matrix'][i][i]/counts[i]) if counts[i] else 'undefined'})" for i in range(4)])
        add_table(['Model'] + list(STAGES), rows)
        add('Paired changes are candidate minus reference, in percentage points. Every candidate-versus-old and matched-seed multitask-versus-stage-only contrast is retained.'); add()
        add_table(['Candidate', 'Reference', 'Paired n', 'Accuracy change; 95% interval', 'Balanced accuracy change; 95% interval', 'Macro F1 change; 95% interval', 'Candidate only correct', 'Reference only correct'], [[labels[r['candidate']], labels[r['reference']], r['n_paired'], *[diff_cell(r['difference'][m]) for m in METRICS], r['candidate_correct_reference_wrong'], r['candidate_wrong_reference_correct']] for r in s['paired_comparisons'].values()])
        add('Condition summaries show the three prespecified seeds. Their standard deviation measures training-run spread and is not a confidence interval.'); add()
        rows = []
        for mode in MODES:
            summary = s['primary_condition_summaries'][mode]
            for metric in METRICS:
                v = summary['metrics'][metric]
                rows.append([mode, summary['n_seeds'], metric, pct(v['mean']), pct(v['sample_sd']), pct(v['min']), pct(v['max'])])
        add_table(['Condition', 'Seeds', 'Metric', 'Mean', 'Sample SD', 'Minimum', 'Maximum'], rows)
        if cohort == 'original100':
            add('Boundary-sensitive accuracy intervals below additionally assume independent images, which is unproven. Degenerate bootstrap intervals above do not prove certainty or equivalence.'); add()
            add_table(['Model', 'Correct / n; exact-binomial sensitivity interval'], [[labels[n], exact_cell(s['models'][n]['accuracy_binomial_sensitivity_analysis'])] for n in ['old_baseline']+names])
    section('Original100 reader comparisons')
    add('Readers form a fixed panel. Comparisons use the same complete cases for each model/reader pair; the panel is not a random sample of neuroradiologists. Missingness and denominators are explicit.'); add()
    reader_rows, contrasts = [], []
    for reader in readers:
        s = a['readers']['paired_comparisons'][reader]
        model = s['models'][reader]
        reader_rows.append([reader, f"{model['point']['correct']}/{model['point']['n']}", len(s['excluded_cases']), *[metric_cell(model, m) for m in METRICS]])
        for r in s['paired_comparisons'].values():
            contrasts.append([labels[r['candidate']], reader, r['n_paired'], *[diff_cell(r['difference'][m]) for m in METRICS], r['candidate_correct_reference_wrong'], r['candidate_wrong_reference_correct']])
    add_table(['Reader', 'Correct / complete n', 'Excluded cases', 'Accuracy; 95% interval', 'Balanced accuracy; 95% interval', 'Macro F1; 95% interval'], reader_rows)
    add_table(['Model', 'Reader', 'Paired n', 'Accuracy change; 95% interval', 'Balanced accuracy change; 95% interval', 'Macro F1 change; 95% interval', 'Model only correct', 'Reader only correct'], contrasts)
    panel = a['readers']['fixed_panel_summary']
    add(f"Fixed panel: {panel['n_fixed_readers']} readers, {panel['common_complete_cases']} common complete images. Reader-case observations are not pooled as independent trials."); add()
    if 'mean_reader_metrics' in panel:
        add_table(['Fixed-panel mean metric', 'Value'], [[m, pct(panel['mean_reader_metrics'][m])] for m in METRICS])
    add('Full original100 case-aligned labels: ' + link(args.evaluation_dir/'original100_case_aligned_old_candidates_readers.csv') + '. Individual prediction CSVs preserve model probabilities.'); add()
    add('<details>'); add('<summary>All 100 reference, model and reader labels</summary>'); add()
    add_table(['Case', 'Reference', 'Old baseline'] + names + readers, [[r['FileName'], r['TrueCategory'], r['old_baseline']] + [r[n] for n in names+readers] for r in aligned])
    add('</details>'); add()
    section('Clinical AD/CN endpoint by source and split')
    add('Clinical AD/CN is distinct from the four-stage image endpoint. Only multitask candidates have a trained clinical head; no clinical old-versus-new improvement is measured. Each source subject contributes one mean-probability prediction. MCI remains prediction-only and is excluded from every binary metric.'); add()
    add_table(['Multitask candidate', 'Clinical training subjects', 'Clinical validation subjects'], [[labels[n], r['config']['training_aux_subjects'], r['config']['validation_aux_subjects']] for n, r in runs.items() if r['selection']['mode'] == 'multitask'])
    for cohort, results in a['clinical'].items():
        add('### ' + esc(cohort)); add()
        add_table(['Candidate', 'All subjects', 'Binary evaluable', 'MCI prediction-only', 'Accuracy correct/n; exact 95% CI', 'AD sensitivity detected/nAD; exact 95% CI', 'CN specificity correct/nCN; exact 95% CI', 'Balanced accuracy', 'Macro F1'], [[labels[n], v['n_subjects'], v['n_binary_evaluable'], v['n_mci_prediction_only'], exact_cell(v['accuracy_exact']), exact_cell(v['AD_sensitivity_exact']), exact_cell(v['CN_specificity_exact']), pct(v['point']['balanced_accuracy']) if v['point'] else 'undefined', pct(v['point']['macro_f1']) if v['point'] else 'undefined'] for n, v in results.items()])
        for n, v in results.items():
            if v['mci_prediction_only']:
                add('MCI descriptive probabilities for ' + esc(labels[n]) + ':'); add()
                add_table(['Source subject', 'P(CN)', 'P(AD)'], [[r['subject_id'], number(r['P_CN'], 6), number(r['P_AD'], 6)] for r in v['mci_prediction_only']])
    add('An external cohort containing only one AD participant cannot establish stable sensitivity, even with perfect observed accuracy. OpenNeuro was inspected during preprocessing preparation, so it is an exploratory external pilot rather than an entirely unseen pipeline validation. Images can encode age and source differences despite age not being a model input.'); add()
    section('Resource cleanup evidence')
    if args.cleanup_json:
        cleanup = read_json(args.cleanup_json)
        add('The supplied cleanup record is reproduced verbatim below. Only explicit recorded deletion verification supports a deletion claim; this report does not contact Runpod or infer absence from a stopped state.'); add()
        add(link(args.cleanup_json) + '; SHA256 `' + digest(args.cleanup_json) + '`.'); add()
        add('```json\n' + json.dumps(cleanup, indent=2, ensure_ascii=False) + '\n```'); add()
    else:
        add('No cleanup metadata was supplied to this report; cloud resource deletion is not verified here.'); add()
    section('Interpretation limits and remaining work')
    for limitation in a['limitations']:
        add('- ' + esc(limitation))
    add('- The auxiliary comparison tests the complete training regimen, including shared-backbone and batch-normalization updates; it does not isolate only the number of added MRI scans.')
    add('- Neither low training loss nor a favorable secondary subset establishes a stronger model overall. Report class-specific regressions and uncertainty; do not select a new winner from held-out results.')
    add('- Dedicated probability calibration improvement has not been assessed by this report. NLL and AUROC in the evaluator evidence are separate from the accuracy-based summaries shown here.')
    add('- Abid\'s ten recommended sources are tracked access candidates; the newly acquired source volumes are Besga and OpenNeuro. This experiment does not imply access to or training on all ten. OASIS-1/2/3 should not be pooled without resolving provider overlap constraints.')
    add()
    section('Supporting outputs')
    figures_dir = getattr(args, 'figures_dir', None)
    if figures_dir:
        require(figures_dir.is_dir(), 'Supplied figure directory is missing')
        for p in sorted(figures_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in {'.png', '.svg', '.pdf'}:
                add('- ' + link(p, 'Companion figure: ' + p.name))
    for p in sorted(args.uncertainty_json.parent.glob('*.csv')):
        add('- ' + link(p))
    for n in ['old_baseline']+names:
        add('- ' + link(args.evaluation_dir/f'{n}_original100_predictions.csv', labels[n] + ' original100 probabilities'))
    add(); add('Report generator SHA256: `' + digest(__file__) + '`.'); add()
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--evaluation-dir', type=Path, required=True)
    p.add_argument('--uncertainty-json', type=Path, required=True)
    p.add_argument('--validation-selection', type=Path, required=True)
    p.add_argument('--cleanup-json', type=Path)
    p.add_argument('--figures-dir', type=Path, help='Optional existing companion figures; never required')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Report output already exists; preserve prior report evidence')
    content = render(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(content)
    print(json.dumps({'status': 'complete', 'output': str(args.output.resolve()), 'sha256': digest(args.output)}))


if __name__ == '__main__':
    main()
