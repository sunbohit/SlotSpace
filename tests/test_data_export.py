"""M0 contracts, including H0 on a real export if one is present."""

import csv
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from slotspace.prime_cache import prepare_prime
from slotspace.stark_io import (
    DOC_SCHEMA, EDGE_SCHEMA, SPLITS, _clean_details, _text_view, _write_edges,
    _write_queries, export_prime,
)

DOC_TYPES = {
    "disease": 17080, "gene_protein": 27671, "molecular_function": 11169,
    "drug": 7957, "pathway": 2516, "anatomy": 14035, "effect_phenotype": 15311,
    "biological_process": 28642, "cellular_component": 4176, "exposure": 818,
}
RELATIONS = {
    "ppi", "carrier", "enzyme", "target", "transporter", "contraindication",
    "indication", "off-label use", "synergistic interaction", "associated with",
    "parent-child", "phenotype absent", "phenotype present", "side effect",
    "interacts with", "linked to", "expression present", "expression absent",
}
SPLIT_COUNTS = {"train": 6162, "val": 2241, "test": 2801, "test-0.1": 280, "human": 98}


def test_details_and_existing_cache_are_preserved(tmp_path):
    assert _clean_details({"_key": 4, "x_id": 5, "a": None, "b": "NaN", "c": "",
                           "aliases": ["foo", "bar"], "zero": 0}) == {"aliases": "foo, bar", "zero": "0"}
    owned = tmp_path / "docs.parquet"
    owned.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="overwrite"):
        export_prime(tmp_path)
    assert owned.read_bytes() == b"existing"
    with pytest.raises(FileNotFoundError, match="--root"):
        prepare_prime(tmp_path)


def test_extra_processed_file_is_rejected_before_upstream_can_open_it(tmp_path):
    processed = tmp_path / "prime" / "processed"
    processed.mkdir(parents=True)
    (processed / "unlisted.pkl").touch()
    with pytest.raises(ValueError, match="unlisted.pkl"):
        prepare_prime(tmp_path)


def test_sample_edges_keep_real_codes_and_both_endpoints(tmp_path):
    skb = SimpleNamespace(edge_index=torch.tensor([[8, 3, 0, 3], [3, 8, 8, 8]]),
                          edge_types=torch.tensor([29, 29, 2, 29]), edge_type_dict={29: "off-label use", 2: "ppi"})
    out = tmp_path / "edges.parquet"
    assert _write_edges(skb, out, np.array([3, 8])) == 3
    table = pq.read_table(out)
    assert table.schema == EDGE_SCHEMA
    assert table.to_pydict() == {"src": [8, 3, 3], "dst": [3, 8, 8], "rel": ["off-label use"] * 3}
    skb.edge_types[0] = 30
    with pytest.raises(ValueError, match="Unknown"):
        _write_edges(skb, tmp_path / "bad.parquet", None)


def test_lookup_matches_coalesced_rows_without_changing_source():
    idx = torch.tensor([[1, 0, 1, 1, 3], [2, 1, 0, 2, 0]])
    coo = torch.sparse_coo_tensor(idx, torch.ones(idx.shape[1]), size=(4, 4))
    skb = SimpleNamespace(sparse_adj_by_type={"r": coo}, node_types=np.zeros(4),
                          get_neighbor_nodes=lambda i, rel: coo[i].coalesce().indices().view(-1).tolist())
    view = _text_view(skb)
    assert [view.get_neighbor_nodes(i, "r") for i in range(4)] == [[1], [0, 2], [], [0]]
    assert view.get_neighbor_nodes is not skb.get_neighbor_nodes


def test_split_files_contain_ids_not_positions_and_partial_answers_are_not_trimmed(tmp_path):
    (tmp_path / "split").mkdir()
    tables = tmp_path / "stark_qa"
    tables.mkdir()
    for name, rows in [
        ("stark_qa", [(42, "line one,\nline two", "[8]"), (7, "other", "[8, 3]")]),
        ("stark_qa_human_generated_eval", [(88, "human", "[8]")]),
    ]:
        with (tables / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(("id", "query", "answer_ids"))
            w.writerows(rows)
    for split in SPLITS[:-1]:
        (tmp_path / "split" / f"{split}.index").write_text("7\n42\n")
    out = tmp_path / "out"
    out.mkdir()
    _write_queries(tmp_path, out, {8})
    assert json.loads((out / "queries_test.jsonl").read_text()) == {
        "qid": 42, "query": "line one,\nline two", "answers": [8],
    }
    assert json.loads((out / "queries_human.jsonl").read_text())["qid"] == 88


@pytest.fixture(scope="module")
def prime_export():
    path = Path(os.environ.get("SLOTSPACE_PRIME_EXPORT", "data/prime"))
    if not path.exists() and "SLOTSPACE_PRIME_EXPORT" not in os.environ:
        pytest.skip("full M0 export unavailable; set SLOTSPACE_PRIME_EXPORT to require H0")
    assert path.is_dir(), f"Missing M0 export: {path}"
    return path


@pytest.fixture(scope="module")
def doc_ids(prime_export):
    docs = pq.read_table(prime_export / "docs.parquet", columns=["doc_id", "doc_type"])
    ids = docs["doc_id"].to_numpy()
    assert len(ids) == 129375 and np.all(ids[:-1] < ids[1:])
    assert Counter(docs["doc_type"].to_pylist()) == DOC_TYPES
    return ids


@pytest.mark.slow
def test_full_documents(prime_export, doc_ids):
    docs = pq.ParquetFile(prime_export / "docs.parquet")
    assert docs.schema_arrow == DOC_SCHEMA
    for batch in docs.iter_batches(batch_size=2048, columns=["name", "text", "details_json"]):
        names, texts, details = batch.columns
        assert not any(col.null_count for col in batch.columns)
        assert all(names.to_pylist()) and all(texts.to_pylist())
        for raw in details.to_pylist():
            obj = json.loads(raw)
            assert isinstance(obj, dict) and all(isinstance(value, str) for value in obj.values())


@pytest.mark.slow
def test_full_edges_and_all_reverse_edges(prime_export, doc_ids):
    path = prime_export / "edges.parquet"
    assert pq.read_schema(path) == EDGE_SCHEMA
    df = pq.read_table(path, read_dictionary=["rel"]).to_pandas()
    assert len(df) == 8100498 and not df.isna().any().any()
    assert set(df["rel"].unique()) == RELATIONS
    assert np.isin(df.src, doc_ids).all() and np.isin(df.dst, doc_ids).all()
    stride = int(doc_ids[-1]) + 1
    for _, group in df.groupby("rel", sort=False, observed=True):
        src, dst = group.src.to_numpy(), group.dst.to_numpy()
        assert np.array_equal(np.sort(src * stride + dst), np.sort(dst * stride + src))


@pytest.mark.slow
def test_full_queries_and_answers(prime_export, doc_ids):
    docs = set(doc_ids.tolist())
    splits = {}
    for split, expected in SPLIT_COUNTS.items():
        with (prime_export / f"queries_{split}.jsonl").open(encoding="utf-8") as src:
            rows = [json.loads(line) for line in src]
        assert len(rows) == expected
        splits[split] = {r["qid"]: r for r in rows}
        assert len(splits[split]) == expected
        for row in rows:
            assert set(row) == {"qid", "query", "answers"}
            assert type(row["qid"]) is int and isinstance(row["query"], str) and row["query"].strip()
            assert isinstance(row["answers"], list) and row["answers"]
            assert all(type(a) is int and a in docs for a in row["answers"])
    train, val, test = (splits[name].keys() for name in ("train", "val", "test"))
    assert not (train & val or train & test or val & test)
    assert all(splits["test"].get(qid) == row for qid, row in splits["test-0.1"].items())


@pytest.mark.slow
def test_text_matches_official_formatter(prime_export, doc_ids):
    root = os.environ.get("SLOTSPACE_STARK_ROOT")
    if not root:
        pytest.skip("set SLOTSPACE_STARK_ROOT to check byte parity against the official formatter")
    from stark_qa import load_skb

    prime = prepare_prime(Path(root))
    kb = load_skb("prime", root=str(prime.parent), download_processed=True)
    rng = np.random.default_rng(13)
    ids = rng.choice(doc_ids, size=100, replace=False).tolist()
    table = pq.read_table(prime_export / "docs.parquet", columns=["doc_id", "text"],
                          filters=[("doc_id", "in", pa.array(ids))])
    assert table.num_rows == len(ids)
    for row in table.to_pylist():
        assert row["text"] == kb.get_doc_info(row["doc_id"], add_rel=True, compact=True)
