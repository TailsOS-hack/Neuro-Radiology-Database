"""Plot every archived epoch unchanged and independently recount test predictions."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/final_review_20260920'
RUNS = {'primary': 'experiments_dedup_regularized',
        'sensitivity': 'experiments_perceptual_regularized'}
TASKS = ['binary', 'tumor', 'dementia', 'eight_class']


def review_metrics(path):
    m = json.loads(path.read_text())
    rows = list(csv.DictReader((path.parent / 'predictions.csv').open()))
    cm = np.zeros((len(m['class_names']), len(m['class_names'])), dtype=int)
    for row in rows:
        true, pred = int(row['true_index']), int(row['pred_index'])
        if row['true_label'] != m['class_names'][true] or row['pred_label'] != m['class_names'][pred]:
            raise ValueError(f'Class mapping mismatch: {path}')
        cm[true, pred] += 1
    correct = int(cm.trace())
    if not (np.array_equal(cm, m['confusion_matrix']) and len(rows) == m['n']
            and correct == m['correct'] and np.isclose(correct / len(rows), m['accuracy'])):
        raise ValueError(f'Prediction/metric mismatch: {path}')
    denom = cm.sum(0) + cm.sum(1)
    macro = float(np.divide(2 * cm.diagonal(), denom, out=np.zeros(len(cm)), where=denom != 0).mean())
    if not np.isclose(macro, m['classification_report']['macro avg']['f1-score']):
        raise ValueError(f'Macro F1 mismatch: {path}')
    return {'n': len(rows), 'correct': correct, 'errors': len(rows)-correct,
            'accuracy': correct/len(rows), 'macro_f1': macro,
            'metrics_path': str(path.relative_to(ROOT)),
            'predictions_sha256': hashlib.sha256((path.parent/'predictions.csv').read_bytes()).hexdigest()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    for cohort, folder in RUNS.items():
        summary[cohort] = {}
        runs = json.loads((ROOT/'training_logs'/folder/'publication_summary.json').read_text())
        for row in runs:
            summary[cohort][row['model']] = review_metrics(ROOT/row['metrics_path'])
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout='constrained')
        for col, task in enumerate(TASKS):
            item = summary[cohort][task]
            path = (ROOT/item['metrics_path']).parent.parent/'history.json'
            h = json.loads(path.read_text())
            epochs = [r['epoch'] for r in h]
            if epochs != list(range(1, len(h)+1)):
                raise ValueError(f'Nonsequential epoch history: {path}')
            for key in ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy']:
                a = np.array([r[key] for r in h])
                if not np.isfinite(a).all() or (a < 0).any() or ('accuracy' in key and (a > 1).any()):
                    raise ValueError(f'Invalid history: {path}: {key}')
            best = max(h, key=lambda r: (r['val_accuracy'], r['epoch']))
            item.update({'epochs': len(h), 'first_epoch': h[0], 'final_epoch': h[-1],
                         'best_validation_epoch_last_tie': best['epoch'],
                         'history_path': str(path.relative_to(ROOT)),
                         'history_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            for ax, metric in zip(axes[:, col], ['loss', 'accuracy']):
                for split, color in [('train', '#2166ac'), ('val', '#d6604d')]:
                    ax.plot(epochs, [r[f'{split}_{metric}'] for r in h], label=split, color=color, marker='.', lw=1.8)
                ax.axvline(best['epoch'], color='#666666', ls=':', lw=1)
                ax.set(xlabel='Epoch', ylabel='Logged loss' if metric == 'loss' else 'Accuracy', ylim=(0, None) if metric == 'loss' else (0, 1.025))
                ax.grid(alpha=.18); ax.legend(frameon=False)
            axes[0, col].set_title(task.replace('_', ' ').title())
        fig.suptitle(f'{cohort.title()} CNN training histories — all saved epochs, unsmoothed\nDotted line: last epoch tied for best validation accuracy; test performance is not used for selection.', fontsize=13)
        for ext in ['png', 'svg']:
            fig.savefig(OUT/f'{cohort}_training_curves.{ext}', dpi=240)
        plt.close(fig)
    (OUT/'training_review.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({c:{t:{k:v for k,v in d.items() if k in ['n','errors','accuracy','epochs','final_epoch']} for t,d in ts.items()} for c,ts in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
