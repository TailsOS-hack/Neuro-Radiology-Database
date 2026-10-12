"""Synthetic model and attribution mechanics; no real MRI efficacy evidence."""
import copy
import numpy as np
import pytest
import torch
from research.dementia_round3.models import ModelConfig, build_model, model_metadata
from research.dementia_round3.explanations import gradcam, randomization_diagnostics, region_controls

torch.set_num_threads(1)


@pytest.mark.parametrize('family', ['multiplanar', 'resnet3d18'])
def test_complete_patient_outputs_gradients_and_strict_reload(family):
    torch.manual_seed(7)
    model = build_model(ModelConfig(kind=family, width=4, dropout=0, slice_chunk=7))
    x = torch.randn(2, 1, 16, 20, 24)
    out = model(x)
    assert out.shape == (2, 2)
    torch.nn.functional.cross_entropy(out, torch.tensor([0, 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    clone = build_model(model.config)
    clone.load_state_dict(model.state_dict(), strict=True)
    model.eval(); clone.eval()
    with torch.no_grad():
        torch.testing.assert_close(clone(x), model(x), rtol=0, atol=0)
        torch.testing.assert_close(model(x[:1]), model(x)[:1], rtol=1e-5, atol=1e-6)
    assert model_metadata(model)['parameters'] > 0


def test_multiplanar_representation_includes_every_full_fov_slice():
    model = build_model(ModelConfig(width=4, dropout=0, slice_chunk=5))
    x = torch.randn(1, 1, 16, 20, 24)
    logits, features, metadata = model.explanation_forward(x)
    assert metadata['slice_counts'] == [16, 20, 24]
    assert [f.shape[0] for f in features] == [16, 20, 24]
    assert logits.shape == (1, 2)


def test_anatomical_normalizer_is_explicit_and_roundtrips():
    model = build_model(ModelConfig(kind='anatomical_logistic'))
    train = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 100
    with pytest.raises(ValueError, match='TRAIN'):
        model(train)
    model.fit_normalizer(train)
    expected = train.mean(0).clone()
    assert model(train).shape == (4, 2)
    model(torch.full_like(train, 10))  # inference never refits normalization
    torch.testing.assert_close(model.mean, expected)
    copy_model = build_model(model.config)
    copy_model.load_state_dict(model.state_dict())
    torch.testing.assert_close(copy_model(train), model(train))
    with pytest.raises(ValueError, match='spatial'):
        gradcam(model, train)


@pytest.mark.parametrize('family', ['multiplanar', 'resnet3d18'])
def test_gradcam_actual_gradient_uncropped_and_preserves_model(family):
    torch.manual_seed(15)
    model = build_model(ModelConfig(kind=family, width=4, dropout=.2, slice_chunk=8))
    model.train()
    model.encoder.stem.eval()
    first = next(model.parameters())
    first.grad = torch.ones_like(first)
    state = copy.deepcopy(model.state_dict())
    modes = [m.training for m in model.modules()]
    x = torch.randn(1, 1, 16, 20, 24)
    result = gradcam(model, x)
    assert result['forward_parity_passed']
    assert result['raw_full_volume_map'].shape == (1, 16, 20, 24)
    assert np.isfinite(result['raw_full_volume_map']).all()
    assert np.all(result['raw_full_volume_map'] >= 0)
    assert len(result['native_maps']) == (3 if family == 'multiplanar' else 1)
    assert result['localization_validity_established'] is False
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert modes == [m.training for m in model.modules()]
    assert torch.equal(first.grad, torch.ones_like(first))
    assert x.grad is None
    with pytest.raises(ValueError, match='widen'):
        gradcam(model, x, parity_tolerance=.1)


def test_randomization_and_region_controls_keep_original_inputs_and_parameters():
    torch.manual_seed(5)
    model = build_model(ModelConfig(kind='resnet3d18', width=4, dropout=0))
    x = torch.randn(1, 1, 16, 16, 16)
    original, before = x.clone(), copy.deepcopy(model.state_dict())
    state = torch.random.get_rng_state().clone()
    result = randomization_diagnostics(model, x)
    assert set(result['randomized']) == {'classifier', 'last_stage', 'all'}
    assert any(r['probability_total_variation'][0] > 1e-5 for r in result['randomized'].values())
    assert torch.equal(state, torch.random.get_rng_state())
    mask = np.zeros((16, 16, 16), bool); mask[:4] = True
    control = region_controls(model, x, {'fixed_edge': mask})
    assert control['regions']['fixed_edge']['voxel_fraction'] == .25
    assert control['causal_disease_localization_established'] is False
    assert torch.equal(x, original)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    with pytest.raises(ValueError, match='boolean'):
        region_controls(model, x, {'invalid': np.ones((16, 16, 16))})


def test_unsupported_and_nonfinite_inputs_rejected():
    with pytest.raises(ValueError):
        build_model(ModelConfig(kind='unknown'))
    model = build_model(ModelConfig(kind='resnet3d18', width=4))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 3, 16, 16))
    x = torch.zeros(1, 1, 16, 16, 16); x[0, 0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        model(x)


@pytest.mark.parametrize('context', [torch.no_grad, torch.inference_mode])
def test_gradcam_works_inside_inference_callers_without_changing_context(context):
    model = build_model(ModelConfig(kind='resnet3d18', width=4, dropout=0)).eval()
    with context():
        volume = torch.randn(1, 1, 16, 16, 16)
        result = gradcam(model, volume)
        assert not torch.is_grad_enabled()
        assert result['forward_parity_passed']
    assert all(p.grad is None for p in model.parameters())
