"""Statistical comparisons must preserve completion and pairing semantics."""

import pytest

from neuro_co.cli.stats import compare_methods


def _table():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        [
            dict(
                problem="tsp",
                kind="explain",
                method=method,
                seed=seed,
                metric="score",
                value=value + seed,
                status="complete",
            )
            for method, value in (("a", 0.8), ("b", 0.5))
            for seed in (0, 1)
        ]
    )


def test_failed_runs_do_not_change_paired_comparison():
    pd = pytest.importorskip("pandas")
    table = _table()
    failed = table.iloc[[0]].assign(value=1000.0, status="failed")
    result = compare_methods(pd.concat([table, failed]), "score", n_bootstrap=20)
    assert len(result) == 1
    assert result.iloc[0]["n"] == 2
    assert result.iloc[0]["delta_mean"] == pytest.approx(0.3)


def test_ambiguous_pairing_requires_explicit_selection():
    pd = pytest.importorskip("pandas")
    table = _table()
    with pytest.raises(ValueError, match="select one campaign"):
        compare_methods(pd.concat([table, table.iloc[[0]]]), "score", n_bootstrap=20)
