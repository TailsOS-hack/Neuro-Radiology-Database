"""Synthetic export guard tests; no real checkpoints, MRI, cloud or Git access."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import sys
import torch

spec=importlib.util.spec_from_file_location('weights_export',Path(__file__).with_name('export_research_weights.py'))
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w);w.torch=torch


class ExportGuards(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.source=self.root/'source.pt';self.destination=self.root/'export.pt'

    def clinical(self):
        obj=dict(model={'weight':torch.tensor([1.0,-0.0]),'counter':torch.tensor(3)},epoch=7,
                 metrics={'n':11,'unit':'patient','nll':.4},protocol_sha256='1'*64,manifest_sha256='2'*64,
                 seed=20261021,architecture='mobilenet_v3_large_features_imagenet_v1_dropout02_linear960x2',
                 selection_metric='unweighted_patient_nll',research_mode='full_fixed_protocol')
        torch.save(obj,self.source)
        record=dict(kind='clinical',seed=20261021,epoch=7,protocol_sha256='1'*64,path=self.source,source_sha256=w.sha(self.source))
        return obj,record

    def stage(self):
        obj={k:0 for k in w.STAGE_KEYS|w.REMOVED}
        obj.update(task='dementia',arch='mobilenet_v3_large',class_names=['a','b','c','d'],image_size=224,
            model_state={'weight':torch.tensor([1.0,-0.0])},aux_state={'weight':torch.tensor([2.0])},epoch=6,round=2,
            config=dict(seed=20261011,mode='stage_only',test_used_for_selection=False,protocol_sha256='1'*64),
            smoke_only=False,optimizer_state={'private_training_state':torch.tensor([9.0])},scheduler_state={},history=[{'epoch':6}])
        torch.save(obj,self.source)
        record=dict(kind='stage',seed=20261011,epoch=6,protocol_sha256='1'*64,path=self.source,source_sha256=w.sha(self.source))
        return obj,record

    def receipt_fixture(self):
        suite=self.root/'suite';suite.mkdir();archive=self.root/'results.tar.gz';payload=b'synthetic payload'
        row=dict(path='kevin_results/checkpoint.pt',size=len(payload),sha256=hashlib.sha256(payload).hexdigest())
        manifest=json.dumps(dict(version=1,files=[row])).encode()
        with tarfile.open(archive,'w:gz') as tar:
            for name,data in [('kevin_results/checkpoint.pt',payload),('kevin_results/ARTIFACT_SHA256.json',manifest)]:
                info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
        (suite/'checkpoint.pt').write_bytes(payload);(suite/'ARTIFACT_SHA256.json').write_bytes(manifest)
        receipt=self.root/'receipt.json';receipt.write_text(json.dumps(dict(archive=str(archive),archive_sha256=w.sha(archive),destination=str(suite),verified_payload_files=1,destination_readback_verified=True)))
        checksum=self.root/'remote.sha256';checksum.write_text(w.sha(archive)+'  results.tar.gz\n')
        return receipt,checksum,suite

    def test_stage_only_removes_three_fields_and_keeps_all_tensor_bytes(self):
        obj,record=self.stage();proof=w.prepare_checkpoint(record,self.destination)
        loaded=torch.load(self.destination,weights_only=True)
        self.assertEqual(set(loaded),w.STAGE_KEYS)
        self.assertEqual(proof['retained_tensors'],2)
        self.assertEqual(proof['removed_top_level_fields'],sorted(w.REMOVED))
        w.exact_tree({k:v for k,v in obj.items() if k not in w.REMOVED},loaded)

    def test_clinical_best_preserves_original_bytes(self):
        _,record=self.clinical();proof=w.prepare_checkpoint(record,self.destination)
        self.assertEqual(self.source.read_bytes(),self.destination.read_bytes())
        self.assertEqual(proof['removed_top_level_fields'],[])

    def test_unknown_or_optimizer_clinical_field_refused(self):
        obj,record=self.clinical();obj['optimizer']={};torch.save(obj,self.source)
        with self.assertRaisesRegex(ValueError,'optimizer/private/unknown'):w.prepare_checkpoint(record,self.destination)
        self.assertFalse(self.destination.exists())

    def test_unknown_stage_field_refused_without_silent_deletion(self):
        obj,record=self.stage();obj['unreviewed_field']='value';torch.save(obj,self.source)
        with self.assertRaisesRegex(ValueError,'Unexpected stage'):w.prepare_checkpoint(record,self.destination)

    def test_private_patient_field_refused(self):
        obj,record=self.clinical();obj['metrics']['patient_ids']=['synthetic_case'];torch.save(obj,self.source)
        with self.assertRaisesRegex(ValueError,'Private/unsupported'):w.prepare_checkpoint(record,self.destination)

    def test_private_path_metadata_refused(self):
        with self.assertRaisesRegex(ValueError,'Private identifier/path'):w.privacy_check({'description':'/private/synthetic/input'})

    def test_bitwise_equality_detects_signed_zero(self):
        with self.assertRaisesRegex(ValueError,'Tensor state changed'):w.exact_tree(torch.tensor([0.0]),torch.tensor([-0.0]))

    def test_nonfinite_tensor_refused(self):
        with self.assertRaisesRegex(ValueError,'Nonfinite'):w.tensor_bytes(torch.tensor([float('nan')]))

    def test_real_verifier_receipt_schema_and_local_readback(self):
        receipt,checksum,suite=self.receipt_fixture();proof=w.verify_retrieval(receipt,checksum,suite,self.root)
        self.assertTrue(proof['destination_readback_verified']);self.assertEqual(proof['verified_payload_files'],1)
        self.assertNotIn('destination',proof);self.assertNotIn('archive',proof)

    def test_receipt_destination_mismatch_refused(self):
        receipt,checksum,suite=self.receipt_fixture();value=json.loads(receipt.read_text());value['destination']=str(self.root);receipt.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'destination/readback'):w.verify_retrieval(receipt,checksum,suite,self.root)

    def test_independent_archive_hash_mismatch_refused(self):
        receipt,checksum,suite=self.receipt_fixture();checksum.write_text('0'*64+'\n')
        with self.assertRaisesRegex(ValueError,'Independent archive'):w.verify_retrieval(receipt,checksum,suite,self.root)

    def test_changed_destination_payload_refused(self):
        receipt,checksum,suite=self.receipt_fixture();(suite/'checkpoint.pt').write_bytes(b'changed result')
        with self.assertRaisesRegex(ValueError,'Retrieved local payload'):w.verify_retrieval(receipt,checksum,suite,self.root)

    def test_existing_different_export_never_overwritten(self):
        self.source.write_bytes(b'new');self.destination.write_bytes(b'old')
        with self.assertRaisesRegex(ValueError,'Existing export differs'):w.publish_identical_or_absent(self.source,self.destination)
        self.assertEqual(self.destination.read_bytes(),b'old')

    def test_existing_identical_export_is_preserved(self):
        self.source.write_bytes(b'same');self.destination.write_bytes(b'same');inode=self.destination.stat().st_ino
        w.publish_identical_or_absent(self.source,self.destination);self.assertEqual(self.destination.stat().st_ino,inode)

    def loaders(self):
        sys.path.insert(0,str(w.HERE.parent/'training'));sys.path.insert(0,str(w.HERE.parent/'round2_clinical'))
        import evaluate_candidates as stage_loader
        import clinical_bag as clinical
        stage_loader.c.imports();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        return stage_loader,clinical

    def test_existing_stage_loader_exact_synthetic_cpu_export_equivalence(self):
        stage_loader,clinical=self.loaders();torch.manual_seed(9123)
        obj,record=self.stage();model,aux=stage_loader.c.network(pretrained=False)
        obj.update(model_state=model.state_dict(),aux_state=aux.state_dict(),class_names=stage_loader.c.STAGES)
        torch.save(obj,self.source);record['source_sha256']=w.sha(self.source)
        w.prepare_checkpoint(record,self.destination)
        proof=w.inference_equivalence(record,self.destination,stage_loader,clinical)
        self.assertTrue(proof['logits_bitwise_equal']);self.assertFalse(proof['stage_auxiliary_head_evaluated'])

    def test_existing_clinical_builder_exact_synthetic_bag_export_equivalence(self):
        stage_loader,clinical=self.loaders();torch.manual_seed(9124)
        obj,record=self.clinical();obj['model']=clinical.build_model().state_dict()
        torch.save(obj,self.source);record['source_sha256']=w.sha(self.source)
        w.prepare_checkpoint(record,self.destination)
        proof=w.inference_equivalence(record,self.destination,stage_loader,clinical)
        self.assertTrue(proof['probabilities_bitwise_equal']);self.assertEqual(proof['clinical_pooling'],'softmax(mean nine logits)')


if __name__=='__main__':unittest.main(verbosity=2)
