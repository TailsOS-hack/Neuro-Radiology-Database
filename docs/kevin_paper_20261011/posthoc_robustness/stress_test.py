#!/usr/bin/env python3
"""Post-hoc fixed-checkpoint 2D image stress audit; never trains or selects.

All scenarios and tolerances are fixed before this audit's inference. The earlier
primary experiment was already completed, so this is expressly post-hoc and
descriptive. It does not simulate an MRI acquisition or prove clinical readiness.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import platform
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import PIL
import torch
import torchvision
from torchvision import transforms

SEED = 20261011
STAGES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
ORDER = ['old_baseline', 'candidate_1', 'candidate_3', 'candidate_5', 'candidate_2', 'candidate_4', 'candidate_6']
SCENARIOS = [
    dict(name='clean', family='clean', value=0),
    dict(name='brightness_0.90', family='brightness', value=0.90),
    dict(name='brightness_1.10', family='brightness', value=1.10),
    dict(name='contrast_0.90', family='contrast', value=0.90),
    dict(name='contrast_1.10', family='contrast', value=1.10),
    dict(name='noise_sd0.01', family='noise', value=0.01),
    dict(name='noise_sd0.02', family='noise', value=0.02),
    dict(name='blur_sigma0.5', family='blur', value=0.5),
    dict(name='blur_sigma1.0', family='blur', value=1.0),
    dict(name='downsample_192', family='downsample', value=192),
    dict(name='downsample_160', family='downsample', value=160),
    dict(name='jpeg_quality90', family='jpeg', value=90),
    dict(name='jpeg_quality70', family='jpeg', value=70),
]
ATOL, RTOL = 1e-5, 1e-4
PINS = {
    'docs/kevin_paper_20261011/training/train_candidate.py': 'cbf67b7c6e842ce35eeef2e0c0a6a0752b865cec3399d7aa58bd5e8ff0e2a5e2',
    'docs/kevin_paper_20261011/training/evaluate_candidates.py': '7ecebb4db2843ab90152ee31d859e03e725d9c2d476637fada1d3b557bf8231d',
    'models/alzheimers_classifier.pt': 'a8cc03aea74f5310c1fc7936c72f94a44e55af11e2ada2a8ba5f8edd6d924db5',
    'training_logs/experiments/kevin_20261011/prepared/stage_manifest.csv': '66a0cd9409d61edd32ee137154f769c749664f255ddebe461cf7683c1f4ba7b9',
    'training_logs/experiments/kevin_20261011/prepared/same_case_test_intersection.csv': '7cf400fe66e8edc9bf86b784976ab1682e1ee8e933ee0d90b15cfd785b36ac05',
    'data/evaluation/ground_truth/radiologist_test_key.csv': '067b3b6bd3f7d1224e50befe7dcb11c43dff878aeb1aac6c6f3f9dd6eabdb9af',
    'data/external/clinical_t1_preprocessed_v1/clinical_slices.csv': '9dc70cad44f40f06dd028fe5e90790d656cf142ce4fb1d6f2f828646719951c1',
}
RESULT_PINS = {
    'validation_selection_before_evaluation.json': '5ba9de51fe339ab6e470de496571430ab6f463a814efc858c228db954241c0c7',
    'frozen_evaluation_all6/evaluation.json': '2ad94f62c20b3505f3585abb6165e75aa4a532e916f41d482f7f7f288f2a56d0',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csv_read(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def check_probs(p):
    if p.ndim != 2 or p.shape[1] != 4 or not np.isfinite(p).all():
        raise ValueError('Invalid probability shape or nonfinite probability')
    if np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(1), 1, atol=2e-6, rtol=0):
        raise ValueError('Probabilities are outside [0,1] or not normalized')


def case_seed(path):
    # Same noise field across models and noise levels; independent of processing order.
    key = f'{SEED}|noise|{path}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big')


def perturb(im, scenario, path):
    family, value = scenario['family'], scenario['value']
    if im.mode != 'RGB' or im.size != (224, 224): raise ValueError('Expected resized RGB224 input')
    if family == 'clean': return im.copy()
    if family == 'brightness': return ImageEnhance.Brightness(im).enhance(value)
    if family == 'contrast': return ImageEnhance.Contrast(im).enhance(value)
    if family == 'noise':
        a = np.asarray(im).astype(np.float32) / 255
        # Spatial additive Gaussian display noise, identical in all RGB channels.
        n = np.random.default_rng(case_seed(path)).normal(0, value, (224, 224, 1))
        return Image.fromarray(np.rint(np.clip(a + n, 0, 1) * 255).astype(np.uint8), 'RGB')
    if family == 'blur': return im.filter(ImageFilter.GaussianBlur(radius=value))
    if family == 'downsample':
        return im.resize((value, value), Image.Resampling.BILINEAR).resize((224, 224), Image.Resampling.BILINEAR)
    if family == 'jpeg':
        stream = io.BytesIO(); im.save(stream, format='JPEG', quality=value, subsampling=0, optimize=False)
        stream.seek(0)
        with Image.open(stream) as decoded: return decoded.convert('RGB')
    raise ValueError(f'Unknown scenario: {family}')


def metrics(y, p, clean):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
    check_probs(p); pred = p.argmax(1); cpred = clean.argmax(1)
    cm = confusion_matrix(y, pred, labels=range(4))
    result = dict(n=len(y), accuracy=float(accuracy_score(y, pred)),
                  balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                  macro_f1=float(f1_score(y, pred, labels=range(4), average='macro', zero_division=0)),
                  nll=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
                  brier_sum=float(np.mean(np.sum((p - np.eye(4)[y]) ** 2, axis=1))),
                  label_flip_count=int((pred != cpred).sum()), label_flip_fraction=float((pred != cpred).mean()),
                  clean_correct_to_wrong=int(((cpred == y) & (pred != y)).sum()),
                  clean_wrong_to_correct=int(((cpred != y) & (pred == y)).sum()),
                  mean_total_variation=float(np.mean(np.sum(abs(p - clean), axis=1) / 2)),
                  confusion_matrix=cm.tolist())
    for k, name in enumerate(STAGES): result[f'recall_{name}'] = float(cm[k, k] / cm[k].sum())
    return result


def prepare(root, results):
    inputs = {}
    for rel, expected in PINS.items():
        p = root / rel
        if sha(p) != expected: raise ValueError(f'Frozen input changed: {rel}')
        inputs[str(p)] = expected
    for rel, expected in RESULT_PINS.items():
        p = results / rel
        if sha(p) != expected: raise ValueError(f'Frozen result changed: {rel}')
        inputs[str(p)] = expected
    selection = json.loads((results / 'validation_selection_before_evaluation.json').read_text())
    metadata = json.loads((results / 'frozen_evaluation_all6/evaluation.json').read_text())
    provenance = json.loads((results / 'evaluation_figures/evaluation_figure_provenance.json').read_text())
    inputs[str(results / 'evaluation_figures/evaluation_figure_provenance.json')] = sha(results / 'evaluation_figures/evaluation_figure_provenance.json')
    if selection['frozen_before_heldout_evaluation'] is not True: raise ValueError('Missing selection freeze')
    if set(selection['candidates']) != set(ORDER) - {'old_baseline'}: raise ValueError('Need all six candidates')
    expected_pairs = {(mode, seed) for mode in ('stage_only', 'multitask') for seed in (20261011, 20261012, 20261013)}
    if {(r['mode'], r['seed']) for r in selection['candidates'].values()} != expected_pairs: raise ValueError('Unexpected mode/seed set')
    checkpoints = {'old_baseline': root / 'models/alzheimers_classifier.pt'}
    identities = {'old_baseline': {'sha256': PINS['models/alzheimers_classifier.pt'], 'mode': 'old_baseline', 'seed': None}}
    for name, record in selection['candidates'].items():
        p = results / f"{record['mode']}_seed{record['seed']}" / 'best.pt'
        if sha(p) != record['checkpoint_sha256'] or metadata['checkpoints'][name]['sha256'] != record['checkpoint_sha256']:
            raise ValueError(f'Checkpoint identity mismatch: {name}')
        checkpoints[name] = p; identities[name] = record; inputs[str(p)] = sha(p)
    if metadata['checkpoints']['old_baseline']['sha256'] != identities['old_baseline']['sha256']: raise ValueError('Baseline metadata mismatch')
    prepared = root / 'training_logs/experiments/kevin_20261011/prepared'
    stage = csv_read(prepared / 'stage_manifest.csv')
    eligible = {r['path']: r for r in stage if r['split'] == 'test' and r['baseline_strict_split'] == 'test'}
    primary = csv_read(prepared / 'same_case_test_intersection.csv')
    if len(primary) != 1058 or len(eligible) != 1058 or {r['path'] for r in primary} != set(eligible): raise ValueError('Primary membership changed')
    for r in primary:
        if any(r[k] != eligible[r['path']][k] for k in ('label', 'group_id', 'split', 'baseline_strict_split', 'pixel_sha256')):
            raise ValueError('Primary row changed')
    rows = [dict(path=r['path'], label=int(r['label']), cohort='internal_same_case', pixel_sha256=r['pixel_sha256']) for r in primary]
    key = csv_read(root / 'data/evaluation/ground_truth/radiologist_test_key.csv')
    if len(key) != 100 or len({r['FileName'] for r in key}) != 100: raise ValueError('Invalid original100 membership')
    rows += [dict(path='data/evaluation/images/' + r['FileName'], label=STAGES.index(r['TrueCategory']), cohort='original100') for r in key]
    if len({r['path'] for r in rows}) != len(rows): raise ValueError('Duplicate audit path')
    archived = {}
    for name in ORDER:
        archived[name] = {}
        for cohort in ('internal_same_case', 'original100'):
            p = results / f'frozen_evaluation_all6/{name}_{cohort}_predictions.csv'
            expected = provenance['inputs_sha256'].get(str(p))
            if not expected or sha(p) != expected: raise ValueError(f'Archived prediction checksum mismatch: {p}')
            inputs[str(p)] = expected
            records = csv_read(p)
            if len({r['path'] for r in records}) != len(records): raise ValueError('Duplicate archived case')
            aligned = {r['path']: r for r in records}; wanted = [r for r in rows if r['cohort'] == cohort]
            if set(aligned) != {r['path'] for r in wanted}: raise ValueError('Archived case set mismatch')
            pvalues = np.array([[float(aligned[r['path']]['P_' + cl]) for cl in STAGES] for r in wanted]); check_probs(pvalues)
            for r, pp in zip(wanted, pvalues):
                rr = aligned[r['path']]
                if rr['TrueCategory'] != STAGES[r['label']] or rr['ModelPrediction'] != STAGES[int(pp.argmax())]: raise ValueError('Archived label/argmax disagreement')
            archived[name][cohort] = pvalues
    return rows, checkpoints, identities, archived, inputs, selection


def plot_results(out, records, identities):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#333333', '#0072B2', '#56B4E9', '#009E73', '#D55E00', '#CC79A7', '#E69F00']
    families = [('brightness', 'Intensity factor', 1), ('contrast', 'Contrast factor', 1), ('noise', 'Gaussian noise SD [0,1]', 0), ('blur', 'Gaussian blur sigma (px)', 0), ('downsample', 'Intermediate pixels per side', 224), ('jpeg', 'JPEG quality', 100)]
    labels = {name: ('Old baseline' if name == 'old_baseline' else f"{identities[name]['mode'].replace('_',' ')} | {identities[name]['seed']}") for name in ORDER}
    lookup = {(r['cohort'], r['model'], r['scenario']): r for r in records}
    for cohort in ('internal_same_case', 'original100'):
        for metric, title, ylab in [('accuracy', 'Accuracy under fixed image stress', 'Accuracy'), ('label_flip_fraction', 'Prediction changes from the local clean reference', 'Fraction changing class')]:
            fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharey=True)
            for ax, (family, xlabel, cleanvalue) in zip(axes.flat, families):
                scenarios = [(cleanvalue, 'clean')] + [(s['value'], s['name']) for s in SCENARIOS if s['family'] == family]
                scenarios.sort()
                for name, color in zip(ORDER, colors):
                    ax.plot([v for v, _ in scenarios], [lookup[(cohort, name, key)][metric] for _, key in scenarios], marker='o', ms=3, color=color, ls='--' if name in ('candidate_2', 'candidate_4', 'candidate_6') else '-', label=labels[name])
                ax.set_title(family.capitalize()); ax.set_xlabel(xlabel); ax.set_ylim(-0.025, 1.025); ax.grid(alpha=.2)
                if family in ('downsample', 'jpeg'): ax.invert_xaxis()
                ax.set_ylabel(ylab)
            handles, legend_labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=3, frameon=False, fontsize=9)
            desc = 'full primary 1058 image rows' if cohort == 'internal_same_case' else 'historically reused original100'
            fig.suptitle(f'POST-HOC DESCRIPTIVE AUDIT | {desc}\n{title}', fontsize=14, weight='bold')
            fig.text(.05, .02, 'Fixed checkpoints; no tuning or reselection. Synthetic 2D image perturbations do not establish independent patient or clinical robustness.\nLines connect only the measured scenario levels; clean comparisons use the current local device, not a new primary evaluation.', fontsize=9)
            fig.subplots_adjust(left=.07, right=.98, top=.78, bottom=.12, hspace=.38, wspace=.2)
            for ext in ('png', 'pdf'): fig.savefig(out / f'{cohort}_{metric}_stress_curves.{ext}', dpi=250)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args(); root = args.root.resolve(); results = args.results.resolve(); out = args.output.resolve()
    if out.exists(): raise ValueError('Use a new output directory to preserve previous evidence')
    if not 1 <= args.batch_size <= 64: raise ValueError('Batch size must be 1..64')
    if args.device == 'mps' and not torch.backends.mps.is_available(): raise ValueError('MPS unavailable; do not silently change device')
    rows, checkpoints, identities, archived, inputs, selection = prepare(root, results)
    sys.path.insert(0, str(root / 'docs/kevin_paper_20261011/training'))
    import train_candidate as c
    import evaluate_candidates as e
    c.imports(); c.seed_all(SEED); device = torch.device(args.device)
    out.mkdir(parents=True); (out / 'predictions').mkdir()
    protocol = dict(label='POST-HOC DESCRIPTIVE FIXED-CHECKPOINT AUDIT', created_utc=datetime.now(timezone.utc).isoformat(),
                    script_sha256=sha(Path(__file__)), scenarios=SCENARIOS, fixed_seed=SEED, models=identities,
                    model_order=ORDER, class_order=STAGES, cohort_sizes={'internal_same_case':1058, 'original100':100},
                    clean_comparison_tolerance={'absolute':ATOL, 'relative':RTOL, 'criterion':'abs(new-old) <= atol + rtol*abs(old), elementwise; mismatches reported, never overwritten'},
                    input_checksums=inputs, training=False, threshold_tuning=False, checkpoint_reselection=False,
                    primary_experiment_changed=False, validation_preference_unchanged=selection['exploratory_representative_candidate'],
                    transform_order='PIL decode RGB -> exact frozen torchvision Resize(224,224, antialias=True) -> fixed stress -> exact frozen ToTensor and ImageNet normalization',
                    noise='NumPy PCG64 per-case SHA256-derived seed; shared spatial noise across RGB channels; same normalized field across noise levels; clip [0,1] and round to uint8',
                    blur='Pillow GaussianBlur radius in pixels; an approximate Gaussian implementation',
                    downsample='Pillow bilinear resize down then bilinear return to224', jpeg='Pillow JPEG encode/decode, subsampling=0, optimize=False',
                    limitations=['Original100 is historically reused and has old training exposure.', 'Primary exact-row test membership does not establish patient independence.', 'Stress parameters were chosen after the primary experiment; this supplement is descriptive and post-hoc.', 'No MRI acquisition physics, patient variation, diagnosis shift, modality shift, prospective setting, or new clinical cohort is simulated.', 'No acceptance threshold or model superiority is inferred from this audit.'],
                    implementation_references=['https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html', 'https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#PIL.ImageFilter.GaussianBlur', 'https://rsna.github.io/claim/checklist.html'])
    dump(out / 'protocol_before_inference.json', protocol)
    environment = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__, pillow=PIL.__version__, device=args.device, batch_size=args.batch_size, training=False, mixed_precision=False)
    dump(out / 'environment.json', environment)
    resize = transforms.Resize((224, 224), antialias=True)
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    base_images=[]; image_records=[]
    for row in rows:
        p=root/row['path']; digest=sha(p)
        with Image.open(p) as im:
            rgb=im.convert('RGB'); pixel=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            if row.get('pixel_sha256') and pixel != row['pixel_sha256']: raise ValueError('Primary decoded image changed')
            base_images.append(resize(rgb))
            image_records.append(dict(**row, file_sha256=digest, decoded_rgb_sha256=pixel, width=rgb.width, height=rgb.height))
    dump(out/'input_image_checksums.json',image_records)
    # A direct frozen-transform equality check guards the clean preprocessing path.
    frozen_ds=c.dataset(rows,root,False,224)
    for i in (0, len(rows)-1):
        if not torch.equal(normalize(base_images[i]), frozen_ds[i][0]): raise ValueError('Clean transform differs from frozen pipeline')
    models={}
    for name in ORDER:
        model,aux,ck=e.load(checkpoints[name],device)
        if int(ck.get('image_size',224)) != 224: raise ValueError('Unexpected input size')
        if name!='old_baseline':
            rec=identities[name]
            if ck['epoch']!=rec['epoch'] or ck['config']['mode']!=rec['mode'] or ck['config']['seed']!=rec['seed']: raise ValueError('Checkpoint metadata differs from frozen identity')
            if ck['config'].get('test_used_for_selection') is not False: raise ValueError('No validation-only provenance')
        model.eval();aux.eval();models[name]=(model,aux);del ck
    masks={cohort:np.array([r['cohort']==cohort for r in rows]) for cohort in ('internal_same_case','original100')}
    y=np.array([r['label'] for r in rows]); clean={}; records=[]; comparisons=[]; timings=[]
    start=time.monotonic()
    for scenario in SCENARIOS:
        ts=time.monotonic()
        tensor=torch.stack([normalize(perturb(im,scenario,row['path'])) for im,row in zip(base_images,rows)])
        if not torch.isfinite(tensor).all(): raise ValueError('Nonfinite transformed inputs')
        for name in ORDER:
            model,aux=models[name]; values=[]
            with torch.inference_mode():
                for lo in range(0,len(rows),args.batch_size):
                    values.append(c.forward(model,aux,tensor[lo:lo+args.batch_size].to(device),'stage').softmax(1).cpu().numpy())
            p=np.concatenate(values).astype(float);check_probs(p)
            if scenario['name']=='clean': clean[name]=p.copy()
            exported=[]
            for row,pp,cp in zip(rows,p,clean[name]):
                rec=dict(cohort=row['cohort'],path=row['path'],TrueCategory=STAGES[row['label']],ModelPrediction=STAGES[int(pp.argmax())],CleanLocalPrediction=STAGES[int(cp.argmax())])
                rec.update({f'P_{cl}':float(v) for cl,v in zip(STAGES,pp)});exported.append(rec)
            csv_write(out/'predictions'/f"{name}__{scenario['name']}.csv",exported)
            for cohort,mask in masks.items():
                result=metrics(y[mask],p[mask],clean[name][mask]); result.update(model=name,cohort=cohort,scenario=scenario['name'],family=scenario['family'],value=scenario['value'])
                records.append(result)
                if scenario['name']=='clean':
                    old=archived[name][cohort];new=p[mask];diff=abs(new-old);close=np.isclose(new,old,atol=ATOL,rtol=RTOL)
                    mismatch=[]
                    for row,np_,op_,dd,ok in zip([r for r in rows if r['cohort']==cohort],new,old,diff,close):
                        if not ok.all() or int(np_.argmax())!=int(op_.argmax()): mismatch.append(dict(path=row['path'],max_abs_difference=float(dd.max()),probabilities_within_tolerance=bool(ok.all()),argmax_matches=bool(np_.argmax()==op_.argmax())))
                    comparisons.append(dict(model=name,cohort=cohort,n=int(mask.sum()),max_absolute_probability_difference=float(diff.max()),mean_absolute_probability_difference=float(diff.mean()),probability_elements_outside_tolerance=int((~close).sum()),cases_outside_tolerance=int((~close).any(1).sum()),argmax_disagreements=int((new.argmax(1)!=old.argmax(1)).sum()),allclose=bool(close.all()),discrepant_cases=mismatch))
            print(json.dumps({'scenario':scenario['name'],'model':name,'images':len(rows),'elapsed_seconds':round(time.monotonic()-start,2)}),flush=True)
        del tensor
        timings.append(dict(scenario=scenario['name'],seconds=time.monotonic()-ts))
        dump(out/'progress.json',dict(completed_scenarios=[r['scenario'] for r in timings],total_scenarios=len(SCENARIOS),elapsed_seconds=time.monotonic()-start))
    scalar=[{k:v for k,v in r.items() if k!='confusion_matrix'} for r in records]
    csv_write(out/'scenario_metrics.csv',scalar);dump(out/'scenario_metrics.json',records)
    dump(out/'clean_archived_comparison.json',dict(tolerance={'atol':ATOL,'rtol':RTOL},old_environment='archived CUDA torch2.8.0; see original environment record',new_environment=environment,results=comparisons,interpretation='Numerical and class differences are preserved; local clean references anchor local stress differences. Archived primary results are unchanged.'))
    plot_results(out,records,identities)
    by={(r['cohort'],r['model'],r['scenario']):r for r in records}
    lines=['# Post-hoc fixed-checkpoint image stress audit','', 'This is an additional descriptive analysis after the primary experiment. No checkpoint, threshold, architecture or training choice was changed. All seven frozen models are reported.', '', 'The 13 fixed scenarios cover clean inputs, modest intensity/contrast shifts, display noise, blur, lower resolution and JPEG artifacts. They do not simulate MRI acquisition physics or establish clinical robustness.', '', '| Cohort | Model | Local clean accuracy | Lowest scenario accuracy | Scenario(s) | Largest class-flip fraction |','|---|---|---:|---:|---|---:|']
    for cohort in masks:
        for name in ORDER:
            rr=[r for r in records if r['cohort']==cohort and r['model']==name];minimum=min(r['accuracy'] for r in rr)
            worst=', '.join(r['scenario'] for r in rr if r['accuracy']==minimum)
            lines.append(f"| {cohort} | {name} | {by[(cohort,name,'clean')]['accuracy']:.4f} | {minimum:.4f} | {worst} | {max(r['label_flip_fraction'] for r in rr):.4f} |")
    lines += ['', 'The minimum above describes this fixed tested grid, not a selected model or an estimated deployment failure rate.', '', '## Clean reproducibility check', '', f"Preset tolerance: absolute {ATOL}, relative {RTOL}. Across all models and cohorts: {sum(r['cases_outside_tolerance'] for r in comparisons)} case-model pairs have at least one probability outside tolerance; {sum(r['argmax_disagreements'] for r in comparisons)} have different predicted classes. Maximum absolute probability difference: {max(r['max_absolute_probability_difference'] for r in comparisons):.8g}.", '', 'See clean_archived_comparison.json for every discrepancy. The local environment differs from the original CUDA environment; original archived predictions remain authoritative for the primary report.', '', '## Interpretation limits', '']
    lines += ['- '+s for s in protocol['limitations']]
    lines += ['', 'No confidence intervals or clinical acceptance thresholds were added. Plotted lines connect only the actual measured levels. Corruptions are shared across models and have fixed per-case seeds.', '', 'Implementation references: [torchvision intensity transforms](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html), [Pillow GaussianBlur](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#PIL.ImageFilter.GaussianBlur), and [CLAIM reporting checklist](https://rsna.github.io/claim/checklist.html).']
    (out/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    # Verify every protected input and image again after inference, before completion.
    for p,digest in inputs.items():
        if sha(p)!=digest: raise ValueError(f'Protected input changed during audit: {p}')
    for row in image_records:
        if sha(root/row['path'])!=row['file_sha256']: raise ValueError('Input image changed during audit')
    manifest=[dict(path=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()]
    dump(out/'output_checksums.json',manifest)
    for row in manifest:
        p=out/row['path']
        if sha(p)!=row['sha256'] or p.stat().st_size!=row['bytes']: raise ValueError('Output readback failed')
    dump(out/'audit_complete.json',dict(status='completed_posthoc_descriptive_audit',finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,total_image_model_scenarios=len(rows)*len(ORDER)*len(SCENARIOS),scenario_timings=timings,output_manifest_sha256=sha(out/'output_checksums.json'),script_sha256=sha(Path(__file__)),protected_input_rechecks_passed=True,output_readbacks_passed=True,models_reselected=False,cloud_resources_used=False))
    print('POSTHOC_AUDIT_COMPLETE',flush=True)


if __name__ == '__main__': main()
