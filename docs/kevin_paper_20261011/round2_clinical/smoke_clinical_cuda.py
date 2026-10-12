#!/usr/bin/env python3
"""Discarded training-only CUDA correctness smoke; never saves model weights.

Runs one step at each prescribed training phase on four Besga TRAIN patients.
Validation/test/external images and metrics cannot enter this smoke. This is not
a research run, cannot resume as one, and emits only correctness diagnostics.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import torch
from torch import nn
import clinical_bag as c


def state_copy(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def run(root,prepared,evidence,research_output):
    root,prepared,evidence,research_output=map(lambda p:Path(p).resolve(),[root,prepared,evidence,research_output])
    if evidence==research_output or evidence.is_relative_to(research_output) or evidence.is_relative_to(root/'models'):
        raise ValueError('Smoke evidence must be separate from research checkpoints')
    if not torch.cuda.is_available():raise ValueError('CUDA is required for this smoke')
    protocol,rows=c.load_prepared(prepared,root,verify_images=False)
    selected=[]
    for label in [0,1]:
        selected.extend(sorted([r for r in rows if r['split']=='train' and r['label']==label],key=lambda r:r['group_key'])[:2])
    if len(selected)!=4 or len({r['group_key'] for r in selected})!=4 or any(r['split']!='train' for r in selected):
        raise ValueError('Smoke must use exactly four distinct training patients')
    for r in selected:
        if [c.pixel_sha(root/p) for p in r['paths']]!=r['pixel_sha256']:raise ValueError('Training pixels changed')
    frozen={str(p):c.sha(p) for p in [Path(c.__file__),prepared/'protocol.json',prepared/'patients.json',root/protocol['initialization']['path']]}
    research_before={str(p):c.sha(p) for p in research_output.rglob('*') if p.is_file()} if research_output.exists() else {}
    torch.set_num_threads(4);device=torch.device('cuda');started=time.monotonic();phases=[]
    for epoch in [1,6]:
        c.seed_all(9000001)
        model=c.build_model(root/protocol['initialization']['path'],initialize_imagenet=True).to(device)
        c.training_mode(model,epoch,protocol['warmup_head_only_epochs'])
        trainable={n for n,p in model.named_parameters() if p.requires_grad}
        bn_names={n for n,m in model.named_modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)}
        for name,p in model.named_parameters():
            expected=name.startswith('head.') or (epoch==6 and any(name.startswith(f'features.{i}.') for i in [14,15,16]))
            if any(name.startswith(n+'.') for n in bn_names):expected=False
            if p.requires_grad!=expected:raise ValueError('Unexpected trainable parameter: '+name)
        if any(m.training for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)):raise ValueError('BatchNorm not frozen')
        before=state_copy(model)
        x=torch.stack([c.preprocess_bag(r,root,True,9000001,epoch) for r in selected]).to(device)
        y=torch.tensor([r['label'] for r in selected],device=device)
        if x.shape!=(4,9,3,224,224) or not torch.isfinite(x).all():raise ValueError('Invalid smoke training bags')
        head=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('head.')]
        feature=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('head.')]
        groups=[dict(params=head,lr=1e-4)]
        if feature:groups.append(dict(params=feature,lr=1e-5))
        opt=torch.optim.AdamW(groups,weight_decay=.001);opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats();t=time.monotonic()
        slice_logits=model(x.reshape(36,3,224,224));logits=c.bag_logits(slice_logits,4)
        if not torch.equal(logits,slice_logits.reshape(4,9,2).mean(1)):raise ValueError('Patient logit aggregation mismatch')
        prob=logits.softmax(1)
        if not torch.isfinite(prob).all() or not torch.allclose(prob.sum(1),torch.ones(4,device=device),atol=1e-6):raise ValueError('Nonfinite patient probabilities')
        loss=nn.functional.cross_entropy(logits,y)
        if not torch.isfinite(loss):raise ValueError('Nonfinite training-only smoke loss')
        loss.backward()
        grad_norms={}
        for name,p in model.named_parameters():
            if p.requires_grad:
                if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('Missing/nonfinite trainable gradient')
                grad_norms[name]=float(p.grad.norm().detach())
            elif p.grad is not None:raise ValueError('Frozen parameter accumulated gradient')
        if sum(v for k,v in grad_norms.items() if k.startswith('head.'))<=0:raise ValueError('Head has no gradient')
        if epoch==6 and sum(v for k,v in grad_norms.items() if k.startswith('features.'))<=0:raise ValueError('Final features have no gradient')
        norm=nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5)
        if not torch.isfinite(norm):raise ValueError('Invalid clipped gradient norm')
        opt.step();torch.cuda.synchronize();step_seconds=time.monotonic()-t
        after=state_copy(model);changed=[k for k in before if not torch.equal(before[k],after[k])]
        if any(k not in trainable for k in changed):raise ValueError('Frozen parameter or running state changed')
        if not any(k.startswith('head.') for k in changed):raise ValueError('Head did not update')
        if epoch==6 and not any(k.startswith('features.') for k in changed):raise ValueError('Final feature blocks did not update')
        if not all(torch.isfinite(v).all() for v in after.values()):raise ValueError('Nonfinite model state after update')
        phases.append(dict(phase='head_only' if epoch==1 else 'final_three_blocks',epoch=epoch,patients=4,slices=36,
                           trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                           trainable_tensors=len(trainable),changed_tensors=len(changed),bn_and_frozen_state_unchanged=True,
                           patient_logit_aggregation_exact=True,finite_nonzero_gradients=True,
                           smoke_training_loss=float(loss.detach()),gradient_norm=float(norm.detach()),
                           peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),step_seconds=step_seconds))
        del model,opt,x,y,slice_logits,logits,prob,loss,before,after
        torch.cuda.empty_cache()
    if any(c.sha(Path(p))!=h for p,h in frozen.items()):raise ValueError('Smoke changed a frozen source file')
    research_after={str(p):c.sha(p) for p in research_output.rglob('*') if p.is_file()} if research_output.exists() else {}
    if research_after!=research_before:raise ValueError('Smoke touched research output')
    report=dict(status='passed_discarded_training_only_cuda_smoke',research_evidence=False,
                gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                protocol_sha256=c.sha(prepared/'protocol.json'),trainer_sha256=c.sha(Path(c.__file__)),
                development_patient_manifest_sha256=c.sha(prepared/'patients.json'),
                selected_train_group_keys_sha256=hashlib.sha256(json.dumps(sorted(r['group_key'] for r in selected)).encode()).hexdigest(),
                training_patients_used=4,validation_patients_used=0,heldout_patients_used=0,
                model_checkpoints_written=0,research_output_unchanged=True,frozen_sources_unchanged=True,
                phases=phases,seconds=time.monotonic()-started,completed_utc=datetime.now(timezone.utc).isoformat())
    evidence.mkdir(parents=True,exist_ok=True)
    target=evidence/'discarded_cuda_smoke.json'
    if target.exists():raise ValueError('Preserve previous smoke evidence; use a new evidence directory')
    c.atomic_json(target,report)
    if json.loads(target.read_text())!=report:raise ValueError('Smoke evidence readback failed')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=c.ROOT);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--evidence',type=Path,default=Path('/workspace/clinical_smoke_discarded'))
    p.add_argument('--research-output',type=Path,default=Path('/workspace/kevin_clinical_results'))
    a=p.parse_args();print(json.dumps(run(a.root,a.prepared,a.evidence,a.research_output),indent=2))


if __name__=='__main__':main()
