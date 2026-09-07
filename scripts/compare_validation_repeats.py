#!/usr/bin/env python3
"""Compare repeat arrays and tabular metrics; exclude platform-dependent PNG/PDF metadata."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.research_validation import read_csv,require,sha,verify_seal


def compare(cal_a,cal_b,exp_a,exp_b):
    result={'status':'pass','rtol':1e-4,'atol':1e-5,'arrays':0,'max_array_absolute_difference':0.,'csv_rows':0,
            'note':'Same protocol, sample, checkpoint identities and device; compare numeric outputs, not image metadata.'}
    for a,b in [(Path(cal_a),Path(cal_b)),(Path(exp_a),Path(exp_b))]:
        verify_seal(a);verify_seal(b)
        pa=json.loads((a/'provenance.json').read_text());pb=json.loads((b/'provenance.json').read_text())
        for key in ['run_id','protocol_sha256','device','checkpoints','source_sha256']:require(pa[key]==pb[key],'Repeat provenance mismatch')
        for p in sorted(a.rglob('*.npy')):
            other=b/p.relative_to(a);require(other.exists(),'Repeat map availability changed')
            x=np.load(p,allow_pickle=False);y=np.load(other,allow_pickle=False)
            require(x.shape==y.shape and np.allclose(x,y,rtol=1e-4,atol=1e-5),'Repeat numerical mismatch: '+str(p.relative_to(a)))
            result['arrays']+=1;result['max_array_absolute_difference']=max(result['max_array_absolute_difference'],float(np.max(np.abs(x-y))))
        require({str(p.relative_to(a)) for p in a.rglob('*.npy')}=={str(p.relative_to(b)) for p in b.rglob('*.npy')},'Repeat map counts changed')
        for p in sorted(a.glob('*.csv')):
            ra=read_csv(p);rb=read_csv(b/p.name);require(len(ra)==len(rb),'Repeat row count changed')
            for x,y in zip(ra,rb):
                require(set(x)==set(y),'Repeat schema changed')
                for key in x:
                    if x[key]==y[key]:continue
                    try:
                        xx=json.loads(x[key]);yy=json.loads(y[key])
                        require(np.allclose(xx,yy,rtol=1e-4,atol=1e-5),'Repeat CSV metric changed')
                    except (json.JSONDecodeError,TypeError):raise ValueError('Repeat categorical value changed: '+key)
                result['csv_rows']+=1
    return result


if __name__=='__main__':print(json.dumps(compare(*map(Path,sys.argv[1:])),indent=2))
