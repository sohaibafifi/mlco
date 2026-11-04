"""Generic thresholds keep pairing, quality gates, and missing debt explicit."""

import math
from typing import Any

import pytest

from neuro_co.aet.analysis import energy_break_even, paired_aet


def test_full_batch_rounding_and_minimum_integer_volume() -> None:
    result = energy_break_even(100, 3, batch_size=16)
    assert result["continuous_instances"] == pytest.approx(100 / 3)
    assert result["minimum_instances"] == 34
    assert result["full_batches"] == 3
    assert result["full_batch_instances"] == 48
    assert 32 * 3 < 100 <= 48 * 3
    assert energy_break_even(96, 3, batch_size=16)["full_batch_instances"] == 32
    assert energy_break_even(0, 3, batch_size=16)["full_batch_instances"] == 0


@pytest.mark.parametrize("saving", [0, -1])
def test_nonpositive_savings_never_produce_a_finite_crossing(saving: float) -> None:
    result = energy_break_even(100, saving)
    assert result["energy_status"] == "infinite"
    assert result["continuous_instances"] is None
    assert result["full_batch_instances"] is None


def test_bootstrap_keeps_correlated_rounds_paired() -> None:
    result = paired_aet(
        [10, 20, 30, 40],
        [13, 23, 33, 43],
        100,
        batch_size=16,
        required_tolerance_pct=2.5,
        tolerances_pct=[1, 2.5, 5],
        seed=12,
    )
    # Independent resampling would turn the large shared round variation into
    # spurious uncertainty. The measured difference is exactly three in each pair.
    assert result["saving_interval_j_per_instance"] == [3, 3]
    assert result["full_batch_instances_interval"] == [48, 48]
    assert result["energy_break_even"]["continuous_instances"] == pytest.approx(100 / 3)
    assert [row["aet_status"] for row in result["aet_by_tolerance"]] == [
        "infeasible",
        "finite",
        "finite",
    ]
    assert result["aet_by_tolerance"][0]["continuous_instances"] is None
    assert result["aet_by_tolerance"][1]["full_batch_instances"] == 48
    assert "conditional_on_observed_debt" in result["uncertainty"]


def test_saving_interval_crossing_zero_is_unidentified_with_unbounded_threshold() -> None:
    result = paired_aet(
        [2, 2],
        [1, 4],
        100,
        required_tolerance_pct=1,
        tolerances_pct=[5],
        bootstrap_replicates=2000,
        seed=9,
    )
    assert result["energy_break_even"]["energy_status"] == "finite"
    assert result["energy_break_even"]["continuous_instances"] == 200
    assert result["energy_status"] == "unidentified"
    assert result["saving_interval_j_per_instance"][0] < 0
    assert result["interval_upper_unbounded"] is True
    assert result["full_batch_instances_interval"][1] is None
    assert result["aet_by_tolerance"][0]["aet_status"] == "unidentified"
    assert result["aet_by_tolerance"][0]["reason"] == "saving_interval_includes_zero"


def test_no_savings_distinct_from_quality_failure() -> None:
    result = paired_aet(
        [4, 4],
        [2, 2],
        100,
        required_tolerance_pct=2,
        tolerances_pct=[1, 5],
        bootstrap_replicates=100,
    )
    assert result["energy_status"] == "infinite"
    assert result["full_batch_instances_interval"] == [None, None]
    assert result["interval_lower_unbounded"] is True
    assert result["interval_upper_unbounded"] is True
    assert [row["aet_status"] for row in result["aet_by_tolerance"]] == ["infeasible", "infinite"]


def test_missing_quality_does_not_erase_energy_crossing_or_assume_feasibility() -> None:
    result = paired_aet([1, 1], [3, 3], 10, tolerances_pct=[5], bootstrap_replicates=50)
    assert result["energy_break_even"]["continuous_instances"] == 5
    row = result["aet_by_tolerance"][0]
    assert row["quality_status"] == row["aet_status"] == "unidentified"
    assert row["continuous_instances"] is None


def test_incomplete_debt_keeps_lower_bound_but_does_not_qualify_aet() -> None:
    result = paired_aet(
        [1, 1],
        [3, 3],
        10,
        batch_size=4,
        debt_complete=False,
        required_tolerance_pct=1,
        tolerances_pct=[0, 5],
        bootstrap_replicates=50,
    )
    assert result["energy_break_even"]["threshold_is_lower_bound"] is True
    assert result["energy_break_even"]["continuous_instances"] == 5
    assert result["full_batch_instances_interval"] == [8, 8]
    invalid, qualified = result["aet_by_tolerance"]
    assert invalid["aet_lower_bound_instances"] is None
    assert qualified["aet_status"] == "unidentified"
    assert qualified["reason"] == "training_debt_incomplete"
    assert qualified["aet_lower_bound_instances"] == 5
    assert qualified["continuous_instances"] is None


def test_bootstrap_reproducibility_and_units() -> None:
    kwargs: dict[str, Any] = dict(required_tolerance_pct=1, tolerances_pct=[5], seed=91)
    result = paired_aet([1, 2, 3], [3, 7, 9], 100, **kwargs)
    assert result == paired_aet([1, 2, 3], [3, 7, 9], 100, **kwargs)
    scaled = paired_aet([1000, 2000, 3000], [3000, 7000, 9000], 100000, **kwargs)
    assert result["energy_break_even"]["continuous_instances"] == pytest.approx(
        scaled["energy_break_even"]["continuous_instances"]
    )
    assert result["full_batch_instances_interval"] == scaled["full_batch_instances_interval"]


@pytest.mark.parametrize(
    "debt,saving,kwargs",
    [
        (-1, 2, {}),
        (math.nan, 2, {}),
        (1, math.inf, {}),
        (True, 2, {}),
        (1, 2, {"batch_size": True}),
        (1, 2, {"batch_size": 1.5}),
        (1, 2, {"batch_size": 0}),
        (1, 2, {"debt_complete": "yes"}),
    ],
)
def test_invalid_scalar_inputs_fail(debt, saving, kwargs) -> None:
    with pytest.raises(ValueError):
        energy_break_even(debt, saving, **kwargs)


@pytest.mark.parametrize(
    "neural,baseline,kwargs",
    [
        ([], [], {}),
        ([1], [2], {}),
        ([1, 2], [3, 4, 5], {}),
        ([1, math.nan], [2, 3], {}),
        ([1, -1], [2, 3], {}),
        ([[1, 2]], [[3, 4]], {}),
        ([True, False], [2, 3], {}),
        ([1, 2], [2, 3], {"bootstrap_replicates": 1}),
        ([1, 2], [2, 3], {"confidence": 1}),
        ([1, 2], [2, 3], {"seed": -1}),
        ([1, 2], [2, 3], {"required_tolerance_pct": math.nan}),
        ([1, 2], [2, 3], {"tolerances_pct": [-1]}),
    ],
)
def test_invalid_paired_inputs_fail(neural, baseline, kwargs) -> None:
    with pytest.raises(ValueError):
        paired_aet(neural, baseline, 100, **kwargs)


def test_decimal_repayment_boundary_does_not_round_up_an_extra_instance() -> None:
    assert 7 * 0.01 == 0.07
    result = energy_break_even(0.07, 0.01)
    assert result["minimum_instances"] == result["full_batch_instances"] == 7


@pytest.mark.parametrize("scale", [0.001, 1, 1000])
@pytest.mark.parametrize("batch_size", [1, 4, 256])
def test_repayment_boundaries_use_actual_savings_without_a_tolerance(
    scale: float, batch_size: int
) -> None:
    saving = 0.01 * scale
    boundary = (7 * batch_size) * saving
    for debt, expected_batches in (
        (math.nextafter(boundary, -math.inf), 7),
        (boundary, 7),
        (math.nextafter(boundary, math.inf), 8),
    ):
        result = energy_break_even(debt, saving, batch_size=batch_size)
        assert result["full_batches"] == expected_batches
        count = result["full_batch_instances"]
        assert (count - batch_size) * saving < debt <= count * saving
        minimum = result["minimum_instances"]
        assert (minimum - 1) * saving < debt <= minimum * saving


@pytest.mark.parametrize("batch_size", [1, 4, 256])
def test_bootstrap_threshold_endpoints_use_the_same_repayment_boundary(batch_size: int) -> None:
    debt = (7 * batch_size) * 0.01
    result = paired_aet([0, 0], [0.01, 0.01], debt, batch_size=batch_size, bootstrap_replicates=100)
    assert result["energy_break_even"]["full_batch_instances"] == 7 * batch_size
    assert result["full_batch_instances_interval"] == [7 * batch_size, 7 * batch_size]


def test_near_percentile_sign_change_cannot_qualify_an_unbounded_aet() -> None:
    result = paired_aet(
        [1, 1, 1],
        [0.999, 2, 2],
        100,
        required_tolerance_pct=1,
        tolerances_pct=[5],
        bootstrap_replicates=100,
        seed=1,
    )
    assert result["bootstrap_nonpositive_saving_fraction"] == 0.03
    assert result["energy_break_even"]["energy_status"] == "finite"
    assert result["saving_interval_j_per_instance"][0] < 0
    assert result["energy_status"] == "unidentified"
    assert result["full_batch_instances_interval"] == [100, None]
    assert result["interval_upper_unbounded"] is True
    assert result["aet_by_tolerance"][0]["aet_status"] == "unidentified"
