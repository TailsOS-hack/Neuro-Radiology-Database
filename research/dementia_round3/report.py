"""Local, reproducible figures from verified predictions and raw histories.

No smoothing, omitted epochs, participant thumbnails or invented clinical data.
The CLI separately verifies external authorization and statistical replay.
"""
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import common
from .statistics import validate_rows


def render_report(rows, metrics, history_paths, destination):
    rows = validate_rows(rows)
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Report output must be new; preserve earlier figures')
    if metrics.get('n_eligible') != len(rows) or set(metrics.get('sites', {})) != {r['site_id'] for r in rows}:
        raise ValueError('Statistics and prediction roster differ')
    histories = []
    for path in history_paths:
        history = common.read_json(path)
        if not history or [r['epoch'] for r in history] != list(range(len(history))):
            raise ValueError('Every raw epoch from zero is required')
        values = [r['clean'][part]['nll'] for r in history for part in ('train', 'tune')]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Invalid raw loss history')
        status = common.read_json(Path(path).parent/'status.json')
        if status.get('status') != 'complete' or type(status.get('seed')) is not int:
            raise ValueError('Completed seed identity is required for raw curves')
        histories.append((Path(path), history, status['seed']))
    if not histories:
        raise ValueError('All selected-seed histories are required')
    destination.mkdir(parents=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    generated = []

    def save(fig, name):
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            target = destination / (name + '.' + extension)
            fig.savefig(target, dpi=160, bbox_inches='tight')
            generated.append({'name': target.name, 'sha256': common.sha256_file(target)})
        plt.close(fig)

    for index, (path, history, seed) in enumerate(histories):
        fig, ax = plt.subplots(figsize=(8, 4))
        for part, color in [('train', '#23658c'), ('tune', '#c56627')]:
            ax.plot([r['epoch'] for r in history], [r['clean'][part]['nll'] for r in history], label=part, color=color)
        ax.set(title=f'Raw patient cross-entropy — seed {seed}', xlabel='Epoch', ylabel='Unsmoothed clean NLL')
        ax.legend(); save(fig, f'loss_final_seed_{index + 1}')

    sites = sorted(metrics['sites'])
    for model in ('old', 'new'):
        confusion_fig, confusion_axes = plt.subplots(len(sites), 1, figsize=(7, max(3, len(sites)*3)), squeeze=False)
        for index, site in enumerate(sites):
            ax = confusion_axes[index, 0]
            matrix = np.zeros((2, 3), dtype=int)
            for row in rows:
                if row['site_id'] == site:
                    column = int(row[model+'_score'] >= metrics['thresholds'][model]) if row[model+'_status'] == 'ok' else 2
                    matrix[row['diagnosis'], column] += 1
            ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0)
            for y in range(2):
                for x in range(3):
                    ax.text(x, y, str(matrix[y, x]), ha='center', va='center', color='black', bbox={'facecolor': 'white', 'alpha': .7, 'edgecolor': 'none'})
            ax.set(xticks=[0, 1, 2], xticklabels=['Predicted CN', 'Predicted AD', 'Abstain / failed'],
                   yticks=[0, 1], yticklabels=['Actual CN', 'Actual AD'], title=f'{model}: {site}; all eligible patients')
        save(confusion_fig, f'{model}_site_confusion')
        fig, axes = plt.subplots(len(sites), 1, figsize=(8, max(3, len(sites)*2.8)), squeeze=False)
        for index, site in enumerate(sites):
            ax = axes[index, 0]
            data = metrics['sites'][site]['models'][model]
            for y, key, label in [(0, 'specificity_cn', 'Specificity'), (1, 'sensitivity_ad', 'Sensitivity')]:
                rate = data[key]
                if rate['estimate'] is not None:
                    point, (lo, hi) = rate['estimate'], rate['exact95']
                    ax.errorbar(point, y, xerr=[[point-lo], [hi-point]], fmt='o', capsize=4)
                ax.text(1.03, y, f"{rate['correct']}/{rate['denominator']}", va='center')
            ax.axvline(.85, color='gray', ls='--', lw=1)
            ax.set(xlim=(0, 1), ylim=(-.5, 1.5), yticks=[0, 1], yticklabels=['Specificity', 'Sensitivity'],
                   title=f"{model}: {site}; coverage {data['coverage']:.1%}")
        save(fig, f'{model}_site_intervals')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, model in zip(axes, ('old', 'new')):
        ax.plot([0, 1], [0, 1], color='gray', ls='--', label='Reference')
        for site in sites:
            accepted = [r for r in rows if r['site_id'] == site and r[model + '_status'] == 'ok']
            xs, ys = [], []
            for bin_index in range(10):
                group = [r for r in accepted if min(9, int(r[model + '_score']*10)) == bin_index]
                if group:
                    xs.append(np.mean([r[model + '_score'] for r in group])); ys.append(np.mean([r['diagnosis'] for r in group]))
            ax.plot(xs, ys, 'o-', label=f'{site} (accepted n={len(accepted)})')
        ax.set(xlim=(0, 1), ylim=(0, 1), title=f'{model}: accepted-only calibration', xlabel='Mean classification score', ylabel='Observed AD fraction')
        ax.legend(fontsize=8)
    save(fig, 'calibration_by_site')
    common.write_json(destination/'metrics.json', metrics)
    provenance = {'prediction_rows_sha256': common.canonical_hash(rows),
                  'histories': [{'sha256': common.sha256_file(p), 'epochs': len(h), 'seed': seed} for p, h, seed in histories],
                  'figures': generated, 'clinical_approval': False, 'loss_smoothing': False}
    common.write_json(destination/'figure_manifest.json', provenance)
    gate = metrics['research_progression_gate']
    (destination/'REPORT.md').write_text(
        '# Round 3 frozen research evaluation\n\n'
        f"Eligible patients: {len(rows)} across {len(sites)} prespecified institutions.\n\n"
        f"Research progression gate: **{'passed' if gate['passed'] else 'not passed'}**. Clinical readiness is not established.\n\n"
        f"Equal-site balanced accuracy: old {metrics['macro_site_balanced_accuracy']['old']}; new {metrics['macro_site_balanced_accuracy']['new']}. "
        f"Difference {metrics['new_minus_old_delta']}; paired 95% interval {metrics['paired95']}.\n\n"
        'All eligible failures and abstentions count as incorrect in primary metrics. Coverage and accepted-only metrics are in metrics.json. '
        'Intervals condition on these fixed institutions. A five-point estimate with a positive lower interval does not prove a true five-point gain.\n\n'
        'Loss curves retain every recorded epoch without smoothing. Calibration plots show accepted predictions only, in fixed 0.1 bins; sparse bins remain descriptive. '
        'No diagnostic probabilities or causal anatomical localization are claimed. Original100 and all previously evaluated cohorts remain historical.\n\n'
        'Before paper submission, complete scanner/demographic/severity subgroups, acquisition stress, repeat-scan and attribution controls, confounding baselines, '
        'the 90-day sensitivity analysis, exclusion flow, CLAIM checklist and clean-environment reproduction. This report does not certify those additional checks.\n')
    return {'status': 'rendered_research_figures', 'figures': len(generated), 'clinical_readiness': False}
