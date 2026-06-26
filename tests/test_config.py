"""Tests for :class:`moe.config.MoEConfig`.

These verify that the config validates every invariant, serialises losslessly and
that the published presets are themselves valid — the config is the single source
of truth for the whole package, so a silent bad value here corrupts everything
downstream.
"""

from __future__ import annotations

import pytest

from moe.config import MoEConfig

# Each case is (field-overrides, substring-expected-in-error). Every invalid
# branch of ``validate`` is covered so a regression that drops a check fails CI.
INVALID_CASES = [
    ({"top_k": 0}, "top_k"),
    ({"num_experts": 4, "top_k": 5}, "cannot exceed num_experts"),
    ({"capacity_factor": 0.5}, "capacity_factor"),
    ({"alpha": -1.0}, "alpha"),
    ({"beta": -0.5}, "beta"),
    ({"d_model": 0}, "d_model"),
    ({"d_ff": 0}, "d_ff"),
    ({"num_experts": 0}, "num_experts"),
    ({"expert_dropout": 1.0}, "expert_dropout"),
    ({"expert_dropout": -0.1}, "expert_dropout"),
    ({"noise_std_init": -1.0}, "noise_std_init"),
    ({"jitter_noise": -1.0}, "jitter_noise"),
    ({"eps": 0.0}, "eps"),
    ({"router_type": "nope"}, "router_type"),
    ({"activation": "nope"}, "activation"),
    ({"dispatch_strategy": "nope"}, "dispatch_strategy"),
    ({"router_type": "switch", "top_k": 2}, "switch"),
]


@pytest.mark.parametrize("overrides, message", INVALID_CASES)
def test_validate_rejects_invalid(overrides: dict[str, object], message: str) -> None:
    """Each invalid field must raise ``ValueError`` naming the problem.

    Why it matters: validation is the only guard between a typo in a config and a
    silently-wrong training run; every constraint must actually fire.
    """
    config = MoEConfig(**overrides)
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_validate_accepts_default() -> None:
    """The default config is valid and ``validate`` returns ``self`` for chaining."""
    config = MoEConfig()
    assert config.validate() is config


@pytest.mark.parametrize(
    "preset",
    [
        MoEConfig.switch_transformer(),
        MoEConfig.mixtral_style(),
        MoEConfig.gpt4_style(),
    ],
)
def test_presets_are_valid(preset: MoEConfig) -> None:
    """Every published preset must pass validation as constructed."""
    assert preset.validate() is preset


@pytest.mark.parametrize(
    "preset",
    [
        MoEConfig(),
        MoEConfig.switch_transformer(),
        MoEConfig.mixtral_style(),
        MoEConfig.gpt4_style(),
    ],
)
def test_to_dict_from_dict_roundtrip(preset: MoEConfig) -> None:
    """``from_dict(to_dict(cfg))`` reconstructs an equal config (lossless)."""
    restored = MoEConfig.from_dict(preset.to_dict())
    assert restored == preset


def test_from_dict_ignores_unknown_keys() -> None:
    """Unknown keys are dropped so newer-saved configs load in older code."""
    data = MoEConfig().to_dict()
    data["a_future_field"] = 1234
    restored = MoEConfig.from_dict(data)
    assert restored == MoEConfig()


def test_from_dict_rejects_non_mapping() -> None:
    """A non-dict argument raises ``TypeError`` with a helpful message."""
    with pytest.raises(TypeError, match="from_dict expects a dict"):
        MoEConfig.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


def test_capacity_no_drop_returns_all_tokens() -> None:
    """With ``drop_tokens=False`` capacity equals the full token count."""
    config = MoEConfig(num_experts=4, top_k=1, drop_tokens=False)
    assert config.capacity(batch_tokens=100) == 100


def test_capacity_with_drop_follows_formula() -> None:
    """Capacity is ``ceil(cf * top_k * tokens / num_experts)`` when dropping."""
    config = MoEConfig(num_experts=4, top_k=1, capacity_factor=1.0, drop_tokens=True)
    # ceil(1.0 * 1 * 100 / 4) == 25.
    assert config.capacity(batch_tokens=100) == 25


def test_tokens_per_expert_and_summary() -> None:
    """The fair-share helper and the one-line summary behave as documented."""
    config = MoEConfig(num_experts=8)
    assert config.tokens_per_expert(batch_tokens=1024) == pytest.approx(128.0)
    assert "MoEConfig(" in config.summary()
