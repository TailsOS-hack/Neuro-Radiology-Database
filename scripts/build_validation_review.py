#!/usr/bin/env python3
"""Single-command versioned review build. Never replaces baseline or a completed version."""
import argparse
import os
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.research_validation import require,sha,write_json,seal,publish,verify_seal
from scripts.check_validation_review import check
from scripts.render_validation_review import render


def run(args):
    require(not args.output.exists() and not args.output.with_suffix('.zip').exists(),'Completed output/archive already exists; choose a new version directory')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.validation-staging-',dir=args.output.parent))
    shutil.copyfile(args.protocol,stage/'protocol.json')
    for component,source,script,extra in [
        ('explainability',args.explainability_source,'scripts/build_explainability_bundle.py',['--followup']),
        ('calibration',args.calibration_source,'scripts/evaluate_calibration.py',[])]:
        if source:
            verify_seal(source);shutil.copytree(source,stage/component)
        else:
            subprocess.run([sys.executable,str(ROOT/script),*extra,'--protocol',str(args.protocol),
                '--checkpoint-dir',str(args.checkpoint_dir),'--device',args.device,'--output',str(stage/component)],check=True)
    # Snapshot the actual source versions named by provenance; refuse mismatched imports.
    for component in ['explainability','calibration']:
        provenance=json.loads((stage/component/'provenance.json').read_text())
        for name,digest in provenance['source_sha256'].items():
            require(sha(ROOT/name)==digest,f'Current source differs from completed component: {name}')
            target=stage/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_validation.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_external_cohort.py'),
        '--manifest',str(ROOT/'tests/fixtures/external/manifest.csv'),'--label-mapping',str(ROOT/'tests/fixtures/external/label_mapping.json'),
        '--internal-manifest',str(ROOT/'tests/fixtures/external/internal_manifest.csv'),'--internal-root',str(ROOT/'tests/fixtures/external'),
        '--validate-only','--output',str(stage/'external_fixture')],check=True)
    if args.repeat_calibration and args.repeat_explainability:
        from scripts.compare_validation_repeats import compare
        write_json(stage/'repeatability.json',compare(args.calibration_source or stage/'calibration',args.repeat_calibration,
                                                     args.explainability_source or stage/'explainability',args.repeat_explainability))
    import importlib.metadata
    import platform
    write_json(stage/'environment.json', {'python':platform.python_version(), 'packages':{
        name:importlib.metadata.version(name) for name in ['numpy','scipy','torch','torchvision','Pillow','matplotlib','reportlab']}})
    extra=stage/'sources/src/experiment_pipeline.py'
    extra.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/'src/experiment_pipeline.py',extra)
    shutil.copyfile(ROOT/'docs/VALIDATION_V2_REPRODUCTION.md',stage/'REPRODUCTION.md')
    shutil.copyfile(ROOT/'docs/EXTERNAL_VALIDATION_READINESS.md',stage/'EXTERNAL_READINESS.md')
    render(stage)
    validation=check(stage,full=args.full,checkpoint_dir=args.checkpoint_dir)
    write_json(stage/'validation_result.json',validation)
    seal(stage);check(stage,full=False)
    env=dict(os.environ,VALIDATION_TEST_BUNDLE=str(stage))
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_validation*.py','-v'],cwd=ROOT,env=env,capture_output=True,text=True)
    require(test.returncode==0,'Completed-artifact tests failed: '+test.stdout+test.stderr)
    (stage/'TEST_RESULTS.txt').write_text(test.stdout+test.stderr)
    seal(stage);check(stage,full=False)
    publish(stage,args.output)
    archive=args.output.with_suffix('.zip')
    require(not archive.exists(),'Archive destination already exists')
    temp=archive.with_suffix('.zip.partial')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(args.output.rglob('*')):
            if path.is_file():z.write(path,str(Path(args.output.name)/path.relative_to(args.output)))
    with zipfile.ZipFile(temp) as z:
        require(z.testzip() is None,'Archive integrity failure')
        for name,digest in json.loads((args.output/'COMPLETE.json').read_text())['files'].items():
            import hashlib
            require(hashlib.sha256(z.read(str(Path(args.output.name)/name))).hexdigest()==digest,'Archive content checksum mismatch')
    temp.replace(archive)
    write_json(archive.with_suffix('.zip.sha256.json'),{'archive':archive.name,'sha256':sha(archive),'run_id':validation['run_id']})
    from scripts.update_validation_manuscript import update
    if args.output.resolve().is_relative_to(ROOT/'docs'):
        update(args.output.resolve())
    print(json.dumps({'output':str(args.output),'archive':str(archive),'validation':validation},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=ROOT/'docs/validation_protocol_v2.json')
    p.add_argument('--output',type=Path,default=ROOT/'docs/review_bundle_v2/validation-v2-20260907')
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    p.add_argument('--full',action='store_true')
    p.add_argument('--calibration-source',type=Path);p.add_argument('--explainability-source',type=Path)
    p.add_argument('--repeat-calibration',type=Path);p.add_argument('--repeat-explainability',type=Path)
    run(p.parse_args())
