#!/usr/bin/env python3
"""Plot all six raw training histories with validation-frozen checkpoint markers.

Reads history.json, config.json, a best.pt checksum, and the frozen selection JSON.
Never loads checkpoint contents, image pixels, held-out labels, or predictions.
The source training histories and raw trainer figures remain untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


MODES = ('stage_only', 'multitask')
SEEDS = (20261011, 20261012, 20261013)
DISPLAY = {'stage_only': 'Stage only', 'multitask': 'Multitask'}
COLORS = {'train': '#2166AC', 'val': '#D55E00', 'selected': '#242424'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def finite_loss(value, description):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{description}: expected finite nonnegative loss')
    return float(value)


def load_runs(run_root, selection_path):
    """Validate provenance before writing a figure. Do not infer a new selection."""
    selection = read_json(selection_path)
    if selection.get('frozen_before_heldout_evaluation') is not True:
        raise ValueError('Require selection frozen before held-out evaluation')
    wanted = {(mode, seed) for mode in MODES for seed in SEEDS}
    records = selection.get('candidates', {})
    if len(records) != len(wanted):
        raise ValueError('Require all six validation-frozen candidates')
    protocol = selection.get('analysis_protocol', {})
    if tuple(protocol.get('planned_modes', ())) != MODES or tuple(protocol.get('planned_seeds', ())) != SEEDS:
        raise ValueError('Frozen plan does not match the six prespecified conditions')
    runs, seen = {}, set()
    for candidate, record in records.items():
        pair = (record.get('mode'), record.get('seed'))
        if pair not in wanted or pair in seen:
            raise ValueError('Unexpected or duplicate frozen condition/seed')
        seen.add(pair)
        mode, seed = pair
        run = Path(run_root) / f'{mode}_seed{seed}'
        config_path, history_path, checkpoint_path = [run / name for name in ('config.json', 'history.json', 'best.pt')]
        config = read_json(config_path)
        if (config.get('mode'), config.get('seed')) != pair:
            raise ValueError(f'{candidate}: run identity differs from frozen selection')
        if config.get('smoke') is not False or config.get('test_used_for_selection') is not False:
            raise ValueError(f'{candidate}: refusing smoke or non-validation-selected run')
        if config.get('stage_manifest_sha256') != selection.get('stage_manifest_sha256') or not selection.get('stage_manifest_sha256'):
            raise ValueError(f'{candidate}: stage manifest differs from frozen selection')
        if mode == 'multitask' and (not selection.get('clinical_manifest_sha256') or config.get('clinical_manifest_sha256') != selection['clinical_manifest_sha256']):
            raise ValueError(f'{candidate}: clinical manifest differs from frozen selection')
        checkpoint_hash = digest(checkpoint_path)
        if checkpoint_hash != record.get('checkpoint_sha256'):
            raise ValueError(f'{candidate}: checkpoint checksum differs from frozen selection')
        history = read_json(history_path)
        if not isinstance(history, list) or not history:
            raise ValueError(f'{candidate}: full nonempty history required')
        if [r.get('epoch') for r in history] != list(range(1, len(history) + 1)):
            raise ValueError(f'{candidate}: raw epochs must be contiguous from epoch 1')
        keys = ['train_stage_loss', 'val_stage_loss']
        if mode == 'multitask':
            keys += ['train_aux_loss', 'val_aux_loss']
        for row in history:
            for key in keys:
                finite_loss(row.get(key), f'{candidate} epoch {row["epoch"]} {key}')
        epoch = record.get('epoch')
        if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= len(history):
            raise ValueError(f'{candidate}: frozen selected epoch absent from history')
        selected_loss = finite_loss(record.get('stage_validation_nll'), f'{candidate} frozen validation loss')
        if not math.isclose(history[epoch - 1]['val_stage_loss'], selected_loss, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f'{candidate}: selected epoch loss differs from frozen metadata')
        runs[pair] = {'candidate': candidate, 'record': record, 'history': history,
                      'inputs': {'directory': str(run.resolve()),
                                 'config_sha256': digest(config_path),
                                 'history_sha256': digest(history_path),
                                 'best_checkpoint_sha256': checkpoint_hash},
                      'epochs_plotted': len(history), 'selected_epoch': epoch}
    return selection, runs


def plot_panels(runs, output, clinical, synthetic):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'semibold', 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    modes = ('multitask',) if clinical else MODES
    fig, axes = plt.subplots(len(modes), len(SEEDS), figsize=(13.2, 4.8 if clinical else 8.0),
                             squeeze=False, sharex=True, sharey=True)
    train_key, val_key = ('train_aux_loss', 'val_aux_loss') if clinical else ('train_stage_loss', 'val_stage_loss')
    max_epoch = max(runs[(m, s)]['epochs_plotted'] for m in modes for s in SEEDS)
    max_loss = max(row[key] for m in modes for s in SEEDS
                   for row in runs[(m, s)]['history'] for key in (train_key, val_key))
    for i, mode in enumerate(modes):
        for j, seed in enumerate(SEEDS):
            ax, run = axes[i, j], runs[(mode, seed)]
            history, selected = run['history'], run['selected_epoch']
            epochs = [r['epoch'] for r in history]
            ax.plot(epochs, [r[train_key] for r in history], color=COLORS['train'], lw=1.7)
            ax.plot(epochs, [r[val_key] for r in history], color=COLORS['val'], lw=1.7)
            ax.axvline(selected, color=COLORS['selected'], lw=1.0, ls=(0, (4, 3)), alpha=.8)
            ax.plot(selected, history[selected - 1][val_key], 'o', color=COLORS['selected'],
                    markerfacecolor='white', markersize=5, markeredgewidth=1.1, zorder=4)
            ax.set_title(f'{DISPLAY[mode]} | seed {seed}\nSelected epoch {selected}; {len(history)} epochs shown', fontsize=10, pad=9)
            ax.set_xlim(.5, max_epoch + .5)
            ax.set_ylim(0, max(max_loss * 1.07, .05))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.grid(axis='y', color='#D8DDE2', alpha=.7, lw=.6)
            ax.set_axisbelow(True)
            if i == len(modes) - 1:
                ax.set_xlabel('Epoch')
            if j == 0:
                ax.set_ylabel('Loss (natural-log units)')
    title = 'Clinical AD/CN auxiliary head: raw loss by seed' if clinical else 'Four-stage dementia head: raw loss by condition and seed'
    if synthetic:
        title = 'SYNTHETIC QA — ' + title
    fig.suptitle(title, fontsize=14, fontweight='semibold', y=.986)
    labels = (['Train: augmented slices', 'Validation: mean-probability subjects', 'Selected using stage validation']
              if clinical else ['Train: augmented images', 'Validation: images', 'Frozen validation-selected epoch'])
    handles = [Line2D([0], [0], color=COLORS['train'], lw=1.8),
               Line2D([0], [0], color=COLORS['val'], lw=1.8),
               Line2D([0], [0], color=COLORS['selected'], lw=1, ls=(0, (4, 3)), marker='o', markerfacecolor='white', markersize=4)]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .926 if clinical else .95),
               ncol=3, frameon=False, fontsize=9.3, handlelength=2.5, columnspacing=2)
    if clinical:
        note = ('Every recorded epoch is shown without smoothing. Training loss uses sampled augmented slices; validation loss uses each subject’s mean of nine slice probabilities.\n'
                'The two curves use different aggregation units, so their gap is not directly comparable. Checkpoints were selected by four-stage validation NLL, not clinical loss.')
        fig.subplots_adjust(left=.065, right=.985, bottom=.24, top=.75, wspace=.16)
    else:
        note = ('Every recorded epoch is shown without smoothing. Losses are the trainer’s unweighted, unsmoothed-label cross-entropy / NLL summaries;\n'
                'training uses augmentation. These are stage-head losses, not the weighted, label-smoothed training objective or the total multitask objective.')
        fig.subplots_adjust(left=.065, right=.985, bottom=.14, top=.84, hspace=.43, wspace=.16)
    fig.text(.065, .025, note, fontsize=8.7, va='bottom', linespacing=1.5)
    stem = 'clinical_loss_all_three_multitask_seeds' if clinical else 'stage_loss_all_six_runs'
    paths = []
    for ext in ('png', 'pdf'):
        path = output / f'{stem}.{ext}'
        fig.savefig(path, dpi=300, metadata={'Creator': 'plot_training_curves.py'} if ext == 'pdf' else {})
        paths.append(path)
    plt.close(fig)
    return paths


def render(run_root, selection_path, output, synthetic=False):
    selection, runs = load_runs(run_root, selection_path)
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new output directory; prior figures are preserved')
    output.mkdir(parents=True)
    paths = plot_panels(runs, output, clinical=False, synthetic=synthetic)
    paths += plot_panels(runs, output, clinical=True, synthetic=synthetic)
    report = {'status': 'synthetic_qa_not_research' if synthetic else 'all_six_raw_histories_rendered',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'script_sha256': digest(__file__), 'selection_path': str(Path(selection_path).resolve()),
              'selection_sha256': digest(selection_path),
              'heldout_data_read': False, 'smoothing': False, 'epoch_filtering': False,
              'y_axis_starts_at_zero': True, 'shared_stage_axes': True, 'shared_clinical_axes': True,
              'selection_source': 'frozen candidates[].epoch; no new checkpoint selection',
              'training_stage_loss': 'Unweighted cross-entropy on augmented stage images, before label smoothing; excludes auxiliary loss.',
              'validation_stage_loss': 'Unweighted NLL on stage validation images.',
              'training_clinical_loss': 'Unweighted cross-entropy on sampled augmented slices.',
              'validation_clinical_loss': 'Subject-level NLL from mean probabilities across nine slices.',
              'clinical_comparison_limit': 'Different slice-versus-subject units; curve gaps do not directly measure generalization gaps.',
              'runs': {r['candidate']: {k: v for k, v in r.items() if k != 'history'} for r in runs.values()},
              'figures': {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size} for p in paths}}
    (output / 'curve_provenance.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True, help='Contains <mode>_seed<seed> directories with full history.json, config.json, and best.pt')
    p.add_argument('--validation-selection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New directory for publication figures and provenance; never overwrite raw plots')
    p.add_argument('--synthetic-qa', action='store_true', help='Prominently label synthetic fixture figures; never use for research histories')
    args = p.parse_args()
    result = render(args.run_root, args.validation_selection, args.output, args.synthetic_qa)
    print(json.dumps({'status': result['status'], 'output': str(args.output.resolve()), 'figures': list(result['figures'])}))


if __name__ == '__main__':
    main()
