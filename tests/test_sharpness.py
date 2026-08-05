from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from vascx.fundus.features import sharpness
from vascx.fundus.features.sharpness import (
    Sharpness,
    _pulse_profile_mu_w,
    fit_bilateral_profile_params,
    fit_bilateral_segment_sigma,
)


def _synthetic_profile(
    *,
    width: float,
    sigma: float,
    background_slope: float = 0.0,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    x = np.arange(41, dtype=np.float64)
    profile = _pulse_profile_mu_w(
        x,
        i_bg=130.0,
        amp=45.0,
        mu=20.0,
        width=width,
        sigma=sigma,
        bg_slope=background_slope,
    )
    if noise_sigma > 0:
        profile += np.random.default_rng(seed).normal(0.0, noise_sigma, x.size)
    return profile


def test_soft_width_prior_recovers_blur_for_gaussian_like_profile() -> None:
    profile = _synthetic_profile(width=2.0, sigma=3.0, noise_sigma=0.8, seed=4)

    fit = fit_bilateral_profile_params(
        profile,
        diameter=2.0,
        smooth_sigma=0.0,
        min_contrast=4.0,
        width_prior_fraction=0.20,
    )

    assert fit is not None
    sigma, _, _, x_left, x_right, kind = fit
    assert kind == "pulse"
    assert sigma == pytest.approx(3.0, abs=0.25)
    assert x_right - x_left == pytest.approx(2.0, abs=0.25)


def test_background_gradient_does_not_inflate_sigma() -> None:
    fitted_sigmas = []
    for background_slope in (-1.0, -0.5, 0.0, 0.5, 1.0):
        profile = _synthetic_profile(
            width=2.0,
            sigma=3.0,
            background_slope=background_slope,
            noise_sigma=0.8,
            seed=4,
        )
        fit = fit_bilateral_profile_params(
            profile,
            diameter=2.0,
            min_contrast=4.0,
            width_prior_fraction=0.20,
        )

        assert fit is not None
        assert fit[-1] == "pulse"
        fitted_sigmas.append(fit[0])

    assert fitted_sigmas == pytest.approx([3.0] * 5, abs=0.25)
    assert np.ptp(fitted_sigmas) < 0.05


def test_width_prior_is_soft_for_resolved_profile() -> None:
    profile = _synthetic_profile(width=8.0, sigma=1.2)

    fit = fit_bilateral_profile_params(
        profile,
        diameter=6.0,
        smooth_sigma=0.0,
        min_contrast=4.0,
        width_prior_fraction=0.20,
    )

    assert fit is not None
    sigma, _, _, x_left, x_right, kind = fit
    assert kind == "pulse"
    assert sigma == pytest.approx(1.2, abs=0.05)
    assert x_right - x_left == pytest.approx(8.0, abs=0.05)


def test_segment_sigma_still_fits_profiles_separately(monkeypatch) -> None:
    profiles = np.arange(27, dtype=np.float64).reshape(3, 9)
    fitted_sigmas = iter([1.0, 4.0, 2.0])
    calls = []

    def fake_profile_sigma(profile, **kwargs):
        calls.append((profile.copy(), kwargs))
        return next(fitted_sigmas)

    monkeypatch.setattr(sharpness, "fit_bilateral_profile_sigma", fake_profile_sigma)

    result = fit_bilateral_segment_sigma(
        profiles,
        diameter=5.0,
        width_prior_fraction=0.25,
    )

    assert result == 2.0
    assert len(calls) == len(profiles)
    assert all(call[1]["diameter"] == 5.0 for call in calls)
    assert all(call[1]["width_prior_fraction"] == 0.25 for call in calls)


def test_width_prior_fraction_must_be_positive() -> None:
    with pytest.raises(ValueError, match="width_prior_fraction"):
        Sharpness(width_prior_fraction=0.0)


def test_profile_validity_mask_excludes_optic_disc() -> None:
    retina_mask = np.ones((15, 15), dtype=bool)
    disc_mask = np.zeros_like(retina_mask)
    disc_mask[7, 7] = True
    layer = SimpleNamespace(
        retina=SimpleNamespace(
            mask=retina_mask,
            disc=SimpleNamespace(mask=disc_mask),
        )
    )
    feature = Sharpness()

    valid_mask = feature._profile_validity_mask(layer)

    assert retina_mask[7, 7]
    assert not valid_mask[7, 7]
    assert feature._profile_outside_bounds(
        origin=np.array([7.0, 4.0]),
        direction=np.array([0.0, 1.0]),
        half_len=4,
        bounds_mask=valid_mask,
    )
    assert not feature._profile_outside_bounds(
        origin=np.array([3.0, 4.0]),
        direction=np.array([0.0, 1.0]),
        half_len=4,
        bounds_mask=valid_mask,
    )
