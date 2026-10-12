"""Synthetic private-suite migration tests; no trained model/MRI/cloud actions."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import migrate_completed_clinical as m


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name).resolve()
        self.root=self.base/'repo';self.root.mkdir();self.suite=self.base/'original';self.suite.mkdir()
        self.dest=self.base/'destination';self.evidence=self.base/'evidence';self.output=self.base/'migration.tar.gz'
        (self.root/'frozen.py').write_bytes(b'# synthetic source\n')
        self.pins=patch.object(m,'PINS',{'frozen.py':m.sha(self.root/'frozen.py')});self.pins.start();self.addCleanup(self.pins.stop)
        self.checked=dict(status='verified_complete_converged_no_inference',selections=[dict(seed=s,checkpoint_path=f'clinical_seed{s}/best.pt') for s in m.SEEDS])
        self.gate=patch.object(m,'verify_scientific_suite',return_value=self.checked);self.mock_gate=self.gate.start();self.addCleanup(self.gate.stop)
        self.payload={'suite_status.json':b'{"status":"synthetic"}', 'validation_selection_before_any_clinical_test.json':b'{}'}
        for seed in m.SEEDS:
            self.payload[f'clinical_seed{seed}/best.pt']=f'synthetic best {seed}'.encode()
            self.payload[f'clinical_seed{seed}/last.pt']=f'synthetic optimizer {seed}'.encode()
            self.payload[f'clinical_seed{seed}/epoch000_validation.json']=b'{"synthetic":true}'
        self.original_archive=self.base/'full_original.tar.gz';self.receipt=self.base/'receipt.json';self.checksum=self.base/'independent.sha256'
        self.full_fixture()

    def full_fixture(self):
        rows=[]
        for n,value in sorted(self.payload.items()):
            p=self.suite/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(value)
            rows.append(dict(path='kevin_results/'+n,size=len(value),sha256=hashlib.sha256(value).hexdigest()))
        manifest=m.encoded(dict(version=1,files=rows));(self.suite/'ARTIFACT_SHA256.json').write_bytes(manifest)
        with tarfile.open(self.original_archive,'w:gz') as tar:
            for n,value in [(r['path'],self.payload[str(Path(r['path']).relative_to('kevin_results'))]) for r in rows]+[('kevin_results/ARTIFACT_SHA256.json',manifest)]:
                m.add(tar,n,io.BytesIO(value),len(value))
        h=m.sha(self.original_archive)
        self.receipt.write_bytes(m.encoded(dict(archive=str(self.original_archive),archive_sha256=h,destination=str(self.suite),verified_payload_files=len(rows),destination_readback_verified=True)))
        self.checksum.write_text(h+'  full_original.tar.gz\n')

    def build(self):return m.build(self.root,self.suite,self.receipt,self.checksum,self.output)
    def install(self):return m.install(self.output,m.sha(self.output),self.root,self.dest,self.evidence)

    def test_roundtrip_only_three_last_files_omitted_all_scientific_bytes_preserved(self):
        report=self.build();self.assertEqual(report['omitted_last_checkpoints'],3)
        installed=self.install();self.assertTrue(installed['unchanged_verifier_passed'])
        for n,value in self.payload.items():
            self.assertEqual((self.suite/n).read_bytes(),value)
            if n in m.OMIT:self.assertFalse((self.dest/n).exists())
            else:self.assertEqual((self.dest/n).read_bytes(),value)
        self.assertFalse((self.dest/'ARTIFACT_SHA256.json').exists())
        self.assertEqual((self.evidence/'original_artifact_manifest.json').read_bytes(),(self.suite/'ARTIFACT_SHA256.json').read_bytes())
        self.assertEqual(self.mock_gate.call_count,3)

    def test_missing_original_last_checkpoint_refused(self):
        del self.payload[next(iter(m.OMIT))];self.full_fixture()
        with self.assertRaisesRegex(ValueError,'all three last'):self.build()
        self.assertFalse(self.output.exists())

    def test_changed_original_optimizer_file_blocks_before_packaging(self):
        (self.suite/next(iter(m.OMIT))).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'Full original payload'):self.build()
        self.assertFalse(self.output.exists())

    def test_wrong_independent_archive_checksum_refused(self):
        self.checksum.write_text('0'*64+'\n')
        with self.assertRaisesRegex(ValueError,'retrieval receipt'):self.build()

    def test_original_suite_gate_failure_blocks_copy(self):
        self.mock_gate.side_effect=ValueError('synthetic incomplete suite')
        with self.assertRaisesRegex(ValueError,'incomplete'):self.build()
        self.assertFalse(self.output.exists())

    def test_existing_destination_not_overwritten(self):
        self.build();self.dest.mkdir();(self.dest/'owned.txt').write_text('keep')
        with self.assertRaisesRegex(ValueError,'new suite'):self.install()
        self.assertEqual((self.dest/'owned.txt').read_text(),'keep')

    def test_destination_verifier_mismatch_preserves_diagnostics_without_success_receipt(self):
        self.build();self.mock_gate.return_value=dict(self.checked,unexpected=True)
        with self.assertRaisesRegex(ValueError,'verification differs'):self.install()
        self.assertTrue(self.dest.exists());self.assertFalse(self.evidence.exists())

    def test_changed_frozen_source_refused_before_install(self):
        self.build();(self.root/'frozen.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'frozen clinical development'):self.install()
        self.assertFalse(self.dest.exists())

    def test_payload_tamper_detected_against_manifest(self):
        self.build();replacement=self.base/'bad.tar.gz'
        with tarfile.open(self.output,'r:gz') as src,tarfile.open(replacement,'w:gz') as dst:
            for member in src:
                value=src.extractfile(member).read()
                if member.name.endswith('/best.pt'):value=b'tampered'
                m.add(dst,member.name,io.BytesIO(value),len(value))
        with self.assertRaisesRegex(ValueError,'payload bytes changed'):m.verify_archive(replacement,m.sha(replacement))

    def test_link_and_traversal_members_refused(self):
        self.build()
        for kind in ['symlink','traversal']:
            replacement=self.base/(kind+'.tar.gz')
            with tarfile.open(self.output,'r:gz') as src,tarfile.open(replacement,'w:gz') as dst:
                for member in src:
                    value=src.extractfile(member).read();m.add(dst,member.name,io.BytesIO(value),len(value))
                member=tarfile.TarInfo(m.TOP+('../outside' if kind=='traversal' else 'link'))
                if kind=='symlink':member.type=tarfile.SYMTYPE;member.linkname='/outside'
                dst.addfile(member)
            with self.assertRaises(ValueError):m.verify_archive(replacement,m.sha(replacement))


if __name__=='__main__':unittest.main(verbosity=2)
