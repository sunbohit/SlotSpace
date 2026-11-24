import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from slotspace.retrieve import BM25Index, BaselineRetriever, DenseIndex, minmax, topk


def test_bm25_scores_postings_and_empty_query():
    bm25 = BM25Index(["running shoes", "red shoes", "running red cap"])
    positions, values = bm25.search(["run run"], 20)
    all_scores, postings = bm25.scores("run")
    assert positions.shape == (1, 3) and postings == 2
    assert np.allclose(values[0], 2 * all_scores[positions[0]])
    assert set(positions[0, :2]) == {0, 2}
    for query in ("the and", "notincorpus"):
        scores, postings = bm25.scores(query)
        assert not scores.any() and postings == 0
    assert topk(np.array([1.0, 2.0, 2.0, 2.0]), 2).tolist() == [1, 2]


def test_bm25_cache_is_reusable_but_not_for_a_changed_corpus(tmp_path):
    docs, cache = tmp_path / "docs.parquet", tmp_path / "index"
    pq.write_table(pa.table({"text": ["running shoes", "blue hat"]}), docs)
    first = BM25Index.cached(docs, cache).search(["running"], 1)
    second = BM25Index.cached(docs, cache).search(["running"], 1)
    assert np.array_equal(first[0], second[0]) and np.allclose(first[1], second[1])
    pq.write_table(pa.table({"text": ["a different corpus"]}), docs)
    with pytest.raises(ValueError, match="settings differ"):
        BM25Index.cached(docs, cache)


def test_dense_block_search_and_hybrid_scores_every_candidate():
    bm25 = BM25Index(["red red red hat", "red blue hat", "blue boot"])
    docs = np.array([[0, 1], [1, 0], [-1, 0]], dtype=np.float32)
    dense = DenseIndex(docs)
    queries = np.array([[1, 0], [0, 1]], dtype=np.float32)
    pos, scores = dense.search(queries, 2, block_size=1)
    assert pos.tolist() == [[1, 0], [0, 1]]
    assert np.allclose(scores, [[1, 0], [1, 0]])
    retriever = BaselineRetriever(bm25, dense, k_l=1, k_d=1, final_k=100)
    pos, scores, counters = retriever.search("red", "hybrid", queries[0], lambda_l=0.5)
    assert set(pos) == {0, 1} and counters["candidates"] == 2 and counters["dense_searches"] == 1
    lexical, _ = bm25.scores("red")
    union = np.array([0, 1])
    expected = minmax(dense.scores(queries[0])[union]) + 0.5 * minmax(lexical[union])
    assert np.allclose(scores, expected[pos])
    assert lexical[1] > 0  # dense-only candidate must retain its lexical score
