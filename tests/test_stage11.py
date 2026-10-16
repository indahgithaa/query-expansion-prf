import json

from scripts.prepare_stage11_generation import CONFIGS, SCENARIOS, prepare


def test_prepare_generation_input(tmp_path):
    output = tmp_path / "gen_inputs.jsonl"
    summary = prepare(output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert summary["rows"] == 1730
    assert summary["observations"] == 346
    assert summary["qids"] == 173
    assert len({(row["id"], row["configuration"]) for row in rows}) == 1730
    assert {row["configuration"] for row in rows} == set(CONFIGS)
    assert {row["scenario"] for row in rows} == set(SCENARIOS)

    oracle = [row for row in rows if row["configuration"] == "ORACLE"]
    assert all(row["chunk_ids"] == [row["gold_chunk_id"], None, None] for row in oracle)
    assert all(row["chunk_1"] and not row["chunk_2"] and not row["chunk_3"] for row in oracle)


def test_experimental_grid_is_fixed():
    assert len(CONFIGS) * len(SCENARIOS) == 10
    assert CONFIGS[-1] == "ORACLE"
