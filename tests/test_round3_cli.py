"""Entrypoint rejection tests; no patient data or model training."""
from pathlib import Path
import pytest
from research.dementia_round3 import cli
from research.dementia_round3 import common


def test_empty_real_inventory_cannot_create_authorization(tmp_path, capsys):
    out = tmp_path/'authorization.json'
    assert cli.main(['freeze-development', '--output', str(out)]) == 2
    assert not out.exists()
    assert 'no_eligible_patients_acquired' in capsys.readouterr().err


def test_external_labels_cannot_be_omitted_or_changed():
    row = {'patient_id': 'synthetic', 'partition': 'external', 'site_id': 'A', 'diagnosis': 0}
    auth = {'manifest_rows': [row]}
    assert cli._verify_predictions([row], auth, 'external')
    for rows in ([], [row, row], [{**row, 'diagnosis': 1}], [{**row, 'site_id': 'B'}]):
        with pytest.raises(ValueError):
            cli._verify_predictions(rows, auth, 'external')


def test_analysis_does_not_run_on_linux_gpu(monkeypatch):
    monkeypatch.setattr(cli.platform, 'system', lambda: 'Linux')
    with pytest.raises(ValueError, match='local Mac'):
        cli._local_analysis()


def test_parse_declares_concrete_pipeline_commands():
    parser = cli.build_parser()
    for command in ('prepare', 'predict-new', 'predict-paired', 'report', 'freeze-external'):
        assert command in parser.format_help()


def test_report_rejects_predictions_from_another_checkpoint(tmp_path):
    artifacts, provenance = {}, {"preprocessing_hash": "prep"}
    for model in ('new', 'old'):
        path = tmp_path/(model+'.json')
        common.write_json(path, {'checkpoints': [{'seed': 1, 'sha256': model}], 'preprocessing_hash': 'prep'})
        artifacts[model+'_model_bundle'] = common._bind(path)
        provenance[model+'_checkpoint_sha256'] = [model]
    selection = {'protocol_hash': 'protocol', 'artifacts': artifacts}
    payload = {'protocol_hash': 'protocol', 'model_bundles': {k: artifacts[k+'_model_bundle'] for k in ('new', 'old')}, 'model_provenance': provenance}
    cli._verify_prediction_models(payload, selection)
    payload['model_provenance'] = {**provenance, 'new_checkpoint_sha256': ['different']}
    with pytest.raises(ValueError, match='provenance mismatch'):
        cli._verify_prediction_models(payload, selection)
