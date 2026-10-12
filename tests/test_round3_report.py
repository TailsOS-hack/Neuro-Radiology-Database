"""Synthetic figure pipeline verification, never a clinical result."""
import json
import pytest
from research.dementia_round3.report import render_report
from research.dementia_round3.statistics import analyze


def fixture(tmp_path):
    rows = [{'patient_id': f'synthetic_{i}', 'site_id': 'A' if i < 4 else 'B', 'diagnosis': i % 2,
             'old_score': .4, 'new_score': .6 if i % 2 else .2, 'old_status': 'ok', 'new_status': 'ok'} for i in range(8)]
    rows[0].update(new_score=None, new_status='abstain')
    history = [{'epoch': e, 'clean': {'train': {'nll': .9/(e+1)}, 'tune': {'nll': .8/(e+1)}}} for e in range(3)]
    path = tmp_path/'history.json'; path.write_text(json.dumps(history))
    (tmp_path/'status.json').write_text(json.dumps({'seed': 20261031, 'status': 'complete'}))
    return rows, analyze(rows, bootstrap_replicates=50), path


def test_figure_pipeline_keeps_failures_and_raw_history(tmp_path):
    rows, metrics, history = fixture(tmp_path)
    target = tmp_path/'figures'
    result = render_report(rows, metrics, [history], target)
    assert result['figures'] == 12 and not result['clinical_readiness']
    recorded = json.loads((target/'metrics.json').read_text())
    assert recorded['n_eligible'] == 8 and recorded['sites']['A']['models']['new']['coverage'] == .75
    provenance = json.loads((target/'figure_manifest.json').read_text())
    assert provenance['histories'][0]['epochs'] == 3
    assert provenance['histories'][0]['seed'] == 20261031
    for name in ('loss_final_seed_1.png', 'calibration_by_site.pdf'):
        assert (target/name).stat().st_size > 1000
    assert not list(target.glob('*synthetic*'))  # participant IDs never appear in filenames


def test_missing_raw_epoch_prevents_reassuring_curve(tmp_path):
    rows, metrics, path = fixture(tmp_path)
    history = json.loads(path.read_text()); history.pop(1); path.write_text(json.dumps(history))
    with pytest.raises(ValueError, match='Every raw epoch'):
        render_report(rows, metrics, [path], tmp_path/'figures')
