import json

from neuro_co.scale.benchmark import (
    BenchmarkGridConfig,
    iter_benchmark_records,
    write_benchmark_jsonl,
)


def test_iter_benchmark_records_includes_partition_quality() -> None:
    cfg = BenchmarkGridConfig(
        sizes=(8,),
        methods=("sweep", "morton_refined", "feature_aware"),
        seeds=(0,),
        num_instances=1,
        max_customers=3,
        local_constructor="time_window",
        refinement_score_mode="hybrid",
        refinement_hybrid_shortlist=1,
    )
    records = list(iter_benchmark_records(cfg))

    assert len(records) == 3
    assert {record["config"]["method"] for record in records} == {
        "sweep",
        "morton_refined",
        "feature_aware",
    }
    assert all(record["config"]["local_constructor"] == "time_window" for record in records)
    assert all(record["config"]["refinement_score_mode"] == "hybrid" for record in records)
    assert all(record["config"]["refinement_hybrid_shortlist"] == 1 for record in records)
    assert all("partition_quality" in record for record in records)
    assert all("metadata" in record["partition"] for record in records)
    assert all(
        "cluster_count_ratio_to_load_bound" in record["partition_quality"] for record in records
    )
    assert all("cost_per_route" in record["metrics"] for record in records)
    assert all("routes_per_customer" in record["metrics"] for record in records)
    assert all(record["partition_quality"]["missing_customers"] == 0 for record in records)


def test_write_benchmark_jsonl_writes_one_record_per_run(tmp_path) -> None:
    out = tmp_path / "bench.jsonl"
    cfg = BenchmarkGridConfig(
        sizes=(8,),
        methods=("capacity_sweep",),
        seeds=(0, 1),
        num_instances=1,
        max_customers=3,
    )

    count = write_benchmark_jsonl(out, cfg)
    rows = [json.loads(line) for line in out.read_text().splitlines()]

    assert count == 2
    assert len(rows) == 2
    assert {row["seed"] for row in rows} == {0, 1}
    assert all(row["metrics"]["feasible"] for row in rows)
