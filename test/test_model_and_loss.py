"""Unit tests for atmospheric loss functions and model architectures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_root = project_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from configs.model_config import (
    AtmosphereModelConfig,
    makani_reference_model_config,
    large_e384_l8_ablation_config,
)
from neural_atmosphere_operator.models.loss import (
    ChannelRelativeAtmosphereLoss,
    CombinedAtmosphereLoss,
    LossScaler,
    RolloutLoss,
    SpectralLoss,
    latitude_weighted_l1,
    latitude_weighted_mse,
    makani_auto_channel_weights,
)
from neural_atmosphere_operator.models.model import AtmosphereNeuralOperator


def test_loss_functions():
    b, c, h, w = 2, 25, 32, 64
    pred = torch.randn(b, c, h, w, requires_grad=True)
    target = torch.randn(b, c, h, w)

    # 1. Latitude Weighted MSE
    mse = latitude_weighted_mse(pred, target)
    assert mse.ndim == 0
    assert mse > 0.0
    mse.backward()
    assert pred.grad is not None

    pred.grad.zero_()

    # 2. Latitude Weighted L1
    l1 = latitude_weighted_l1(pred, target)
    assert l1.ndim == 0
    assert l1 > 0.0

    # 3. Channel Relative Loss
    rel_loss_fn = ChannelRelativeAtmosphereLoss()
    scalar_loss, per_ch = rel_loss_fn(pred, target)
    assert scalar_loss.ndim == 0
    assert per_ch.shape == (c,)

    # 4. Spectral Loss
    spec_loss_fn = SpectralLoss(loss_type="l1")
    spec_loss = spec_loss_fn(pred, target)
    assert spec_loss.ndim == 0

    # 5. Combined Atmosphere Loss
    combined_loss_fn = CombinedAtmosphereLoss(spectral_weight=0.1)
    comb_loss = combined_loss_fn(pred, target)
    assert comb_loss.ndim == 0


def test_spectral_l2_obeys_parseval_and_pole_cells_have_area():
    pred = torch.randn(2, 3, 7, 10)
    target = torch.randn_like(pred)
    spectral_mse = SpectralLoss(loss_type="l2")(pred, target)
    torch.testing.assert_close(spectral_mse, (pred - target).square().mean())

    pole_error = torch.zeros(1, 1, 721, 1)
    pole_error[:, :, 0] = 1.0
    assert latitude_weighted_mse(pole_error, torch.zeros_like(pole_error)) > 0.0

    with pytest.raises(ValueError, match="length 721"):
        latitude_weighted_mse(
            pole_error, torch.zeros_like(pole_error), latitudes=[0.0, 1.0]
        )


def test_makani_channel_and_temporal_difference_weighting():
    names = (
        "10m_u_component_of_wind",
        "2m_temperature",
        "temperature@1000hPa",
        "geopotential@500hPa",
    )
    base = torch.tensor([0.1, 1.0, 1.0, 0.5])
    base /= base.sum()
    weights = makani_auto_channel_weights(names)
    torch.testing.assert_close(weights, base)

    temporally_scaled = makani_auto_channel_weights(
        names,
        normalization_stds=[2.0, 4.0, 6.0, 8.0],
        time_difference_stds=[1.0, 2.0, 3.0, 4.0],
    )
    torch.testing.assert_close(temporally_scaled, 2.0 * base)

    prediction = torch.ones(1, 4, 5, 8)
    target = torch.zeros_like(prediction)
    weighted_loss = latitude_weighted_mse(
        prediction,
        target,
        channel_weights=weights,
    )
    assert weighted_loss == pytest.approx(1.0)

    with pytest.raises(ValueError, match="supplied together"):
        makani_auto_channel_weights(names, normalization_stds=[1.0] * 4)


def test_optional_loss_scaler_balances_channel_gradient_norms():
    prediction = torch.randn(2, 3, 5, 8, requires_grad=True)
    coefficients = torch.tensor([1.0, 10.0, 100.0]).view(1, 3, 1, 1)
    (LossScaler()(prediction) * coefficients).sum().backward()
    assert prediction.grad is not None
    norms = prediction.grad.norm(dim=(-2, -1))
    torch.testing.assert_close(norms, norms[:, :1].expand_as(norms))


def test_rollout_loss():
    b, k, c, h, w = 2, 3, 4, 16, 32
    preds = torch.randn(b, k, c, h, w, requires_grad=True)
    targets = torch.randn(b, k, c, h, w)

    rollout_loss_fn = RolloutLoss(discount_factor=0.8)
    loss = rollout_loss_fn(preds, targets)
    assert loss.ndim == 0
    loss.backward()
    assert preds.grad is not None


def test_rollout_loss_uses_explicit_time_axis_when_batch_equals_steps():
    class ElementwiseMSE(nn.Module):
        def forward(self, prediction, target):
            return (prediction - target).square().mean()

    # B == K used to make layout inference ambiguous.
    predictions = torch.zeros(2, 2, 1, 1, 1)
    predictions[:, 0] = 1.0
    predictions[:, 1] = 3.0
    targets = torch.zeros_like(predictions)
    loss = RolloutLoss(base_loss=ElementwiseMSE(), discount_factor=0.5, time_dim=1)(
        predictions, targets
    )
    assert loss == pytest.approx((1.0 + 0.5 * 9.0) / 1.5)

    time_first_loss = RolloutLoss(
        base_loss=ElementwiseMSE(), discount_factor=0.5, time_dim=0
    )(predictions.transpose(0, 1), targets.transpose(0, 1))
    torch.testing.assert_close(time_first_loss, loss)


def test_sfno_twenty_four_year_half_degree_defaults_and_makani_reference():
    config = AtmosphereModelConfig()

    assert config.img_size == (361, 720)
    assert config.in_channels == config.out_channels == 26
    assert config.scale_factor == 2
    internal_height = (config.img_size[0] - 1) // config.scale_factor + 1
    internal_width = config.img_size[1] // config.scale_factor
    retained_modes = int(
        min(internal_height, internal_width // 2)
        * config.hard_thresholding_fraction
    )
    assert (internal_height, internal_width, retained_modes) == (181, 360, 180)
    assert config.embed_dim == 128
    assert config.num_layers == 6
    assert config.activation_function == "gelu"
    assert config.normalization_layer == "instance_norm"
    assert config.mlp_ratio == 2.0
    assert config.operator_type == "driscoll-healy"

    reference = makani_reference_model_config()
    assert reference.embed_dim == 384
    assert reference.num_layers == 8
    assert reference == large_e384_l8_ablation_config()
    assert reference.scale_factor == 2


def test_sfno_grid_must_be_exactly_downsampleable():
    with pytest.raises(ValueError, match="height"):
        AtmosphereModelConfig(img_size=(24, 48), scale_factor=3)
    with pytest.raises(ValueError, match="width"):
        AtmosphereModelConfig(img_size=(25, 49), scale_factor=3)


def test_atmosphere_neural_operator_forward_and_rollout():
    h, w = 13, 24
    config = AtmosphereModelConfig(
        img_size=(h, w),
        in_channels=4,
        out_channels=4,
        scale_factor=3,
        embed_dim=8,
        num_layers=2,
        mlp_ratio=2.0,
    )

    model = AtmosphereNeuralOperator(config)
    x = torch.randn(2, 4, h, w)

    # 1. Forward step
    out = model(x)
    assert out.shape == (2, 4, h, w)
    assert "SphericalFourierNeuralOperator" in type(model.backbone).__name__

    # 2. Autoregressive Rollout
    forecasts = model.rollout(x, steps=3)
    assert len(forecasts) == 3
    for f in forecasts:
        assert f.shape == (2, 4, h, w)

    # Rollout is also the path used for multi-step training and must retain the
    # graph through earlier predictions.
    train_input = torch.randn(1, 4, h, w, requires_grad=True)
    model.rollout(train_input, steps=2)[-1].mean().backward()
    assert train_input.grad is not None
    assert torch.isfinite(train_input.grad).all()


def test_model_rollout_shifts_stacked_history_and_uses_latest_residual():
    config = AtmosphereModelConfig(
        img_size=(13, 24),
        in_channels=8,
        out_channels=4,
        scale_factor=3,
        embed_dim=8,
        num_layers=1,
        mlp_ratio=2.0,
        drop_path_rate=0.2,
    )
    model = AtmosphereNeuralOperator(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    x = torch.randn(2, 8, 13, 24)
    torch.testing.assert_close(model(x), x[:, -4:])
    forecasts = model.rollout(x, steps=3)
    assert len(forecasts) == 3
    for forecast in forecasts:
        torch.testing.assert_close(forecast, x[:, -4:])
