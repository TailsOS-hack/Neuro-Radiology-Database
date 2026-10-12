#!/usr/bin/env python3
"""Export six fixed research checkpoints after independently verified retrieval.

Never trains, reads MRI, reselects models, alters production weights, invokes Git,
or uses cloud resources. Real export must wait for both verified local suites.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STAGE_SEEDS = [20261011, 20261012, 20261013]
CLINICAL_SEEDS = [20261021, 20261022, 20261023]
OUTPUT = 'models/kevin_round2_20261011'
REMOVED = {'optimizer_state', 'scheduler_state', 'history'}
HEX = re.compile(r'[0-9a-f]{64}')
STAGE_KEYS = {'task','arch','class_names','image_size','model_state','aux_state','epoch','round',
              'best_selection_score','selection','stage_validation','stress_validation','config',
              'smoke_only','bad_epochs'}
CLINICAL_KEYS = {'model','epoch','metrics','protocol_sha256','manifest_sha256','seed','architecture',
                 'selection_metric','research_mode'}
torch = None


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def regular(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('Expected regular input: '+str(path))
    for parent in path.parents:
        if parent.is_symlink():raise ValueError('Symlink parent refused: '+str(path))
    return path


def relative(root,name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or str(p)!=name:
        raise ValueError('Unsafe relative path')
    return regular(root/str(p))


def independent_digest(path):
    lines=regular(path).read_text().strip().splitlines()
    if len(lines)!=1:raise ValueError('Expected one independently retrieved archive checksum')
    value=lines[0].split()[0]
    if not HEX.fullmatch(value):raise ValueError('Invalid independent checksum')
    return value


def verify_retrieval(receipt_path,checksum_path,suite,root):
    """Recheck the real results_archive.verify receipt, archive and all local bytes."""
    receipt=read(regular(receipt_path));suite=suite.resolve(strict=True)
    if receipt.get('destination_readback_verified') is not True or Path(receipt.get('destination','')).resolve()!=suite:
        raise ValueError('Retrieval receipt destination/readback mismatch')
    archive=Path(receipt['archive']);archive=archive if archive.is_absolute() else root/archive
    expected=independent_digest(checksum_path)
    if receipt.get('archive_sha256')!=expected or sha(regular(archive))!=expected:
        raise ValueError('Independent archive checksum mismatch')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)) or any(not m.isfile() or m.sparse is not None for m in members):
            raise ValueError('Nonregular or duplicate archived result')
        for n in names:
            p=PurePosixPath(n)
            if p.is_absolute() or '..' in p.parts or '\\' in n or len(p.parts)<2 or p.parts[0]!='kevin_results' or str(p)!=n:
                raise ValueError('Unsafe archived result path')
        raw=tar.extractfile('kevin_results/ARTIFACT_SHA256.json').read()
        manifest=json.loads(raw);rows=manifest.get('files',[])
        if manifest.get('version')!=1 or not rows or len(rows)!=receipt.get('verified_payload_files'):
            raise ValueError('Retrieval receipt payload count mismatch')
        indexed={r['path']:r for r in rows}
        if len(indexed)!=len(rows) or set(names)!=set(indexed)|{'kevin_results/ARTIFACT_SHA256.json'}:
            raise ValueError('Archive/result manifest membership mismatch')
        if regular(suite/'ARTIFACT_SHA256.json').read_bytes()!=raw:raise ValueError('Local artifact manifest changed')
        for n,r in indexed.items():
            local=relative(suite,str(PurePosixPath(n).relative_to('kevin_results')))
            if type(r['size']) is not int or local.stat().st_size!=r['size'] or sha(local)!=r['sha256']:
                raise ValueError('Retrieved local payload changed: '+n)
    return {'receipt_sha256':sha(receipt_path),'independent_checksum_file_sha256':sha(checksum_path),
            'archive_sha256':expected,'verified_payload_files':len(rows),'destination_readback_verified':True}


def frozen_sources(root,freeze):
    pins=freeze.get('source_code_sha256',{})
    if not pins:raise ValueError('Missing frozen source pins')
    for name,value in pins.items():
        if not HEX.fullmatch(value) or sha(relative(root,name))!=value:raise ValueError('Frozen loader/source changed: '+name)


def selected_sources(root,suite,kind,expected_freeze_sha):
    """Validate frozen identities and completion linkage without patient/image reads."""
    name='round2_evaluation_freeze.json' if kind=='stage' else 'clinical_evaluation_freeze.json'
    path=regular(suite/name)
    if not HEX.fullmatch(expected_freeze_sha) or sha(path)!=expected_freeze_sha:raise ValueError('Independent freeze checksum mismatch')
    freeze=read(path);frozen_sources(root,freeze);status=read(regular(suite/'suite_status.json'));records=[]
    if kind=='stage':
        selected_path=regular(suite/'validation_selection_before_any_round2_test.json');selected=read(selected_path)
        if (freeze.get('status')!='frozen_before_any_round2_evaluation' or freeze.get('selection_used_only_validation') is not True
                or status.get('status')!='complete' or status.get('completed_seeds')!=STAGE_SEEDS
                or freeze.get('suite_status_sha256')!=sha(suite/'suite_status.json')
                or freeze.get('validation_selection_sha256')!=sha(selected_path)
                or status.get('selection_sha256')!=sha(selected_path) or not selected.get('complete')
                or not selected.get('selection_used_only_fixed_validation')):raise ValueError('Stage completion/selection chain incomplete')
        psha=sha(regular(suite/'protocol.json'))
        if any(v!=psha for v in [status.get('protocol_sha256'),selected.get('protocol_sha256'),freeze.get('training_protocol_sha256')]):raise ValueError('Stage protocol mismatch')
        expected={f'round2_seed{s}' for s in STAGE_SEEDS}
        if set(freeze['all_three_runs'])!=expected:raise ValueError('Wrong frozen stage models')
        for seed in STAGE_SEEDS:
            item=freeze['all_three_runs'][f'round2_seed{seed}'];folder=suite/f'stage_only_seed{seed}'
            sel=read(regular(folder/'selection.json'));run=read(regular(folder/'run_status.json'))
            if (item['seed']!=seed or item['checkpoint']!=f'stage_only_seed{seed}/best.pt'
                    or sel!=selected['selections'][str(seed)] or run!=status['runs'][str(seed)]
                    or run.get('status')!='complete' or run.get('smoke_only') is not False or sel.get('smoke_only') is not False
                    or sha(regular(folder/'selection.json'))!=item['selection_sha256']
                    or sha(regular(folder/'history.json'))!=item['history_sha256']
                    or item['epoch']!=sel['selected']['epoch'] or item['epoch']!=run.get('selected_epoch')
                    or sel['selected']['selection'].get('eligible') is not True
                    or sel['checkpoint_sha256']!=item['checkpoint_sha256']
                    or run.get('best_checkpoint_sha256')!=item['checkpoint_sha256']):raise ValueError('Stage per-seed selection mismatch')
            if run.get('needs_validation_convergence_review') or status.get('needs_validation_convergence_review'):
                review=freeze.get('validation_convergence_review') or {}
                if (review.get('status')!='approved_after_validation_only_review' or review.get('heldout_outcomes_used') is not False
                        or review.get('suite_status_sha256')!=sha(suite/'suite_status.json')
                        or review.get('training_protocol_sha256')!=psha or set(review.get('reviewed_seeds',[]))!=set(STAGE_SEEDS)):
                    raise ValueError('Stage convergence review absent')
            records.append(dict(kind=kind,seed=seed,path=relative(suite,item['checkpoint']),source_sha256=item['checkpoint_sha256'],epoch=item['epoch'],protocol_sha256=psha))
    else:
        selected_path=regular(suite/'validation_selection_before_any_clinical_test.json');selected=read(selected_path)
        verified=freeze.get('training_verification',{})
        if (freeze.get('status')!='frozen_before_clinical_heldout_inference' or freeze.get('heldout_outcomes_used_for_selection') is not False
                or verified.get('status')!='verified_complete_converged_no_inference'
                or status.get('status')!='complete' or status.get('completed_runs')!=3 or status.get('needs_validation_convergence_review')
                or status.get('selection_sha256')!=sha(selected_path) or verified.get('selection_sha256')!=sha(selected_path)
                or selected.get('held_out_data_used') is not False or selected.get('needs_validation_convergence_review')):
            raise ValueError('Clinical completion/selection chain incomplete')
        if set(freeze['selections'])!={f'clinical_seed{s}' for s in CLINICAL_SEEDS}:raise ValueError('Wrong frozen clinical models')
        if [r['seed'] for r in selected['selections']]!=CLINICAL_SEEDS:raise ValueError('Wrong clinical seed order')
        for key in ['protocol_sha256','manifest_sha256']:
            if any(obj.get(key)!=verified.get(key) for obj in [status,selected]):raise ValueError('Clinical training pin mismatch')
        for row in selected['selections']:
            seed=row['seed'];item=freeze['selections'][f'clinical_seed{seed}'];folder=suite/f'clinical_seed{seed}'
            sel=read(regular(folder/'selection.json'));run=read(regular(folder/'run_status.json'))
            expected_path=f'clinical_seed{seed}/best.pt'
            if (row.get('run_dir')!=folder.name or row.get('checkpoint')!='best.pt' or item.get('checkpoint_path')!=expected_path
                    or sel!={k:v for k,v in row.items() if k!='run_dir'}
                    or item!={**sel,'checkpoint_path':expected_path}
                    or run.get('status')!='complete' or any(run.get(k)!=v for k,v in sel.items())
                    or sel.get('stop_reason')!='validation_plateau' or sel.get('research_mode')!='full_fixed_protocol'
                    or sel.get('held_out_data_used') is not False or sel.get('needs_validation_convergence_review')):
                raise ValueError('Clinical per-seed selection mismatch')
            if any(row.get(k)!=verified.get(k) for k in ['protocol_sha256','manifest_sha256']):raise ValueError('Clinical per-seed pins differ')
            for filename,key in [('history.json','history_sha256'),('selected_validation.json','selected_validation_sha256'),('selected_train_clean.json','selected_train_clean_sha256')]:
                if sha(regular(folder/filename))!=row[key]:raise ValueError('Clinical selected evidence changed')
            records.append(dict(kind=kind,seed=seed,path=relative(suite,expected_path),source_sha256=row['checkpoint_sha256'],epoch=row['epoch'],protocol_sha256=row['protocol_sha256']))
    for r in records:
        if not HEX.fullmatch(r['source_sha256']) or sha(r['path'])!=r['source_sha256']:raise ValueError('Frozen selected checkpoint changed')
    return records


def tensor_bytes(tensor):
    if not torch.is_tensor(tensor) or tensor.layout!=torch.strided:raise ValueError('Only dense tensor states supported')
    value=tensor.detach().cpu().contiguous()
    if value.is_floating_point() and not torch.isfinite(value).all():raise ValueError('Nonfinite checkpoint state')
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def exact_tree(a,b):
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.dtype!=b.dtype or a.shape!=b.shape or tensor_bytes(a)!=tensor_bytes(b):raise ValueError('Tensor state changed')
        return 1
    if isinstance(a,dict):
        if not isinstance(b,dict) or list(a)!=list(b):raise ValueError('Metadata/state keys changed')
        return sum(exact_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        if type(a)!=type(b) or len(a)!=len(b):raise ValueError('Metadata sequence changed')
        return sum(exact_tree(x,y) for x,y in zip(a,b))
    if type(a)!=type(b) or a!=b:raise ValueError('Metadata changed')
    return 0


def privacy_check(value,path=()):
    if torch.is_tensor(value):tensor_bytes(value);return
    if isinstance(value,dict):
        for key,item in value.items():
            if not isinstance(key,str) or key.lower() in {'subject_id','subject_ids','group_key','group_keys','patient_id','patient_ids','patients','subjects','path','paths','source_path','image_paths'}:
                raise ValueError('Private/unsupported metadata field: '+'.'.join(path+(str(key),)))
            privacy_check(item,path+(key,))
    elif isinstance(value,(tuple,list)):
        for item in value:privacy_check(item,path)
    elif isinstance(value,str):
        if value.startswith(('/','file:','~')) or re.search(r'\bHV\d{3}\b|\bsubj_\d{2}\b|/Users/|root@|\.ssh/',value):raise ValueError('Private identifier/path in checkpoint metadata')
    elif value is not None and not isinstance(value,(bool,int,float)):raise ValueError('Unsupported checkpoint metadata type')
    elif isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite checkpoint metadata')


def prepare_checkpoint(record,temp):
    source=record['path'];obj=torch.load(source,map_location='cpu',weights_only=True)
    if not isinstance(obj,dict):raise ValueError('Checkpoint wrapper required')
    if record['kind']=='stage':
        if set(obj)!=STAGE_KEYS|REMOVED:raise ValueError('Unexpected stage checkpoint fields; review required')
        if (obj['epoch']!=record['epoch'] or obj['round']!=2 or obj['config'].get('seed')!=record['seed']
                or obj['smoke_only'] is not False or obj['config'].get('mode')!='stage_only'
                or obj['config'].get('test_used_for_selection') is not False
                or obj['config'].get('protocol_sha256')!=record['protocol_sha256']):raise ValueError('Stage checkpoint identity mismatch')
        public={k:v for k,v in obj.items() if k not in REMOVED}
        privacy_check(public)
        # BytesIO gives a stable archive root rather than a temporary filename.
        buf=io.BytesIO();torch.save(public,buf);temp.write_bytes(buf.getvalue());removed=sorted(REMOVED)
    else:
        if set(obj)!=CLINICAL_KEYS:raise ValueError('Clinical best has optimizer/private/unknown fields; preserve and review')
        if (obj['epoch']!=record['epoch'] or obj['seed']!=record['seed'] or obj['research_mode']!='full_fixed_protocol'
                or obj['selection_metric']!='unweighted_patient_nll' or obj['protocol_sha256']!=record['protocol_sha256']
                or obj['architecture']!='mobilenet_v3_large_features_imagenet_v1_dropout02_linear960x2'):raise ValueError('Clinical checkpoint identity mismatch')
        public=obj;privacy_check(public);shutil.copyfile(source,temp);removed=[]
    reread=torch.load(temp,map_location='cpu',weights_only=True);count=exact_tree(public,reread)
    if record['kind']=='clinical' and sha(temp)!=record['source_sha256']:raise ValueError('Clinical original bytes changed')
    return {'removed_top_level_fields':removed,'retained_tensors':count,'all_retained_tensors_bitwise_equal':True,
            'retained_metadata_exactly_equal':True,'contains_clinical_individual_records':False}


def inference_equivalence(record,exported,stage_loader,clinical):
    generator=torch.Generator(device='cpu').manual_seed(2026101209)
    x=torch.randn((18 if record['kind']=='clinical' else 3,3,224,224),generator=generator)
    outputs=[]
    for path in [record['path'],exported]:
        if record['kind']=='stage':
            model,aux,_=stage_loader.load(path,torch.device('cpu'));model.eval();aux.eval()
        else:
            model=clinical.build_model();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'],strict=True);model.eval()
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        with torch.inference_mode():
            if record['kind']=='stage':logits=stage_loader.c.forward(model,aux,x,'stage')
            else:logits=clinical.bag_logits(model(x),2,9)
            probabilities=logits.softmax(1)
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite synthetic output')
        exact_tree(before,dict(model.state_dict()));outputs.append((logits.clone(),probabilities.clone()))
        del model
    exact_tree(outputs[0],outputs[1])
    return {'device':'cpu','generator_seed':2026101209,'input_shape':list(x.shape),'input_sha256':hashlib.sha256(tensor_bytes(x)).hexdigest(),
            'input':'synthetic standard-normal tensors; no MRI','logits_bitwise_equal':True,'probabilities_bitwise_equal':True,
            'max_absolute_logit_difference':0.0,'max_absolute_probability_difference':0.0,'model_state_unchanged_by_eval':True,
            'clinical_pooling':'softmax(mean nine logits)' if record['kind']=='clinical' else None,
            'stage_auxiliary_head_evaluated':False}


def publish_identical_or_absent(source,destination):
    if destination.exists() or destination.is_symlink():
        if sha(regular(destination))!=sha(source):raise ValueError('Existing export differs; never overwrite')
    else:os.link(source,destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    for kind in ['stage','clinical']:
        for name in ['suite','retrieval-receipt','archive-sha256-file']:parser.add_argument('--'+kind+'-'+name,type=Path,required=True)
        parser.add_argument('--'+kind+'-freeze-sha256',required=True)
    args=parser.parse_args();root=args.root.resolve(strict=True)
    if root!=ROOT:raise ValueError('Run from the containing audited repository')
    baseline=regular(root/'models/alzheimers_classifier.pt');baseline_sha=sha(baseline)
    receipts={};records=[]
    for kind in ['stage','clinical']:
        suite=getattr(args,kind+'_suite').resolve(strict=True)
        receipts[kind]=verify_retrieval(getattr(args,kind+'_retrieval_receipt'),getattr(args,kind+'_archive_sha256_file'),suite,root)
        records+=selected_sources(root,suite,kind,getattr(args,kind+'_freeze_sha256'))
    # Import only after independently pinned files and all six source bytes pass.
    sys.path.insert(0,str(HERE.parent/'training'));sys.path.insert(0,str(HERE.parent/'round2_clinical'))
    import evaluate_candidates as stage_loader
    import clinical_bag as clinical
    global torch
    torch=clinical.torch;stage_loader.c.imports();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    output=root/OUTPUT
    if output.is_symlink() or (output.exists() and not output.is_dir()):raise ValueError('Unsafe export destination')
    for parent in output.parents:
        if parent.is_symlink():raise ValueError('Symlink export parent')
    names={f'{r["kind"]}_seed{r["seed"]}.pt' for r in records}|{'CHECKPOINTS.json'}
    if output.exists() and {p.name for p in output.iterdir()}-names:raise ValueError('Unexpected existing export payload')
    with tempfile.TemporaryDirectory(prefix='.kevin-round2-export-',dir=root/'models') as td:
        temp=Path(td);exports=[]
        for record in records:
            filename=f'{record["kind"]}_seed{record["seed"]}.pt';path=temp/filename
            proof=prepare_checkpoint(record,path);synthetic=inference_equivalence(record,path,stage_loader,clinical)
            exports.append(dict(kind=record['kind'],seed=record['seed'],selected_epoch=record['epoch'],file=filename,
                source_checkpoint_sha256=record['source_sha256'],export_sha256=sha(path),export_bytes=path.stat().st_size,
                training_protocol_sha256=record['protocol_sha256'],**proof,synthetic_cpu_equivalence=synthetic))
        manifest=dict(version=1,purpose='Research-only fixed selected inference checkpoints; not clinical deployment',
            stage_class_order=stage_loader.c.STAGES,clinical_class_order=['CN','AD'],clinical_threshold=0.5,
            stage_auxiliary_head='Retained bit-for-bit for source compatibility; not used or clinically evaluated',
            stage_removed_fields=sorted(REMOVED),clinical_original_best_bytes_preserved=True,
            frozen_selection_sha256={k:getattr(args,k+'_freeze_sha256') for k in ['stage','clinical']},retrieval=receipts,
            export_source_sha256=sha(Path(__file__)),torch_version=str(torch.__version__),cpu_threads=1,
            checkpoints=exports,real_image_inference_performed=False,production_baseline_unchanged=True,
            limitations=['Synthetic equality proves export fidelity only, not medical validity.','All test/outcome and privacy limitations of the frozen studies apply.','Full optimizer/history training states remain in private verified archives.'])
        (temp/'CHECKPOINTS.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        # Every prospective byte is ready before any destination mutation.
        for name in names:
            existing=output/name
            if existing.exists() or existing.is_symlink():
                if sha(regular(existing))!=sha(temp/name):raise ValueError('Existing export differs: '+name)
        for record in records:
            if sha(record['path'])!=record['source_sha256']:raise ValueError('Source changed during export')
        if sha(baseline)!=baseline_sha:raise ValueError('Production baseline changed')
        output.mkdir(exist_ok=True)
        for name in sorted(names-{'CHECKPOINTS.json'}):publish_identical_or_absent(temp/name,output/name)
        publish_identical_or_absent(temp/'CHECKPOINTS.json',output/'CHECKPOINTS.json')
        for name in names:
            if sha(regular(output/name))!=sha(temp/name):raise ValueError('Export destination readback failed')
    if sha(baseline)!=baseline_sha:raise ValueError('Production baseline changed after export')
    print(json.dumps({'status':'six_research_checkpoints_exported_and_verified','output':str(output),
                      'CHECKPOINTS_sha256':sha(output/'CHECKPOINTS.json'),'production_baseline_unchanged':True},indent=2))


if __name__=='__main__':main()
