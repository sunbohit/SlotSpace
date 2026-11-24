import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from slotspace.embed import Embedder, context_vectors, topic_vectors


class RecordingModel:
    prompts = {"query": "instruction"}

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(kwargs)
        return np.tile(np.arange(1, 1025, dtype=np.float32), (len(texts), 1))


def test_prompts_truncation_and_immutable_single_pass_topic_cache(tmp_path):
    model = RecordingModel()
    encoder = Embedder.__new__(Embedder)
    encoder.model = model
    encoder.metadata = lambda: {"model": "fake", "max_seq_length": 512}
    encoder.queries(["q"], 128)
    assert model.calls[-1]["prompt_name"] == "query"
    encoder.queries(["segment"], 128, prompt="custom Query:")
    assert model.calls[-1]["prompt"] == "custom Query:"
    docs, cache = tmp_path / "docs.parquet", tmp_path / "vectors"
    pq.write_table(pa.table({"text": ["a", "b", "c"]}), docs)
    before = len(model.calls)
    full, cut = topic_vectors(docs, cache, encoder, batch_size=2)
    assert len(model.calls) == before + 2 and model.calls[-1]["prompt"] == ""
    a, b = np.load(full), np.load(cut)
    assert a.dtype == b.dtype == np.float16 and (a.shape, b.shape) == ((3, 1024), (3, 256))
    assert np.allclose(np.linalg.norm(b.astype(np.float32), axis=1), 1, atol=1e-3)
    assert np.allclose(b, Embedder._cut(a, 256), atol=1e-3)
    topic_vectors(docs, cache, encoder, batch_size=1)
    assert len(model.calls) == before + 2
    pq.write_table(pa.table({"text": ["changed"]}), docs)
    with pytest.raises(ValueError, match="settings differ"):
        topic_vectors(docs, cache, encoder)
    with pytest.raises(ValueError, match="zero"):
        Embedder._cut(np.zeros((1, 1024)), 256)


def test_context_rows_remain_aligned_and_never_receive_query_prompts(tmp_path):
    encoder = Embedder.__new__(Embedder)
    encoder.model = RecordingModel()
    encoder.metadata = lambda: {"model": "fake"}
    facts = tmp_path / "facts.parquet"
    pq.write_table(pa.table({"row": [0, 1], "ctx_text": ["field: x | [disease]", "[drug]"]}), facts)
    path = context_vectors(facts, tmp_path / "context", encoder, batch_size=1)
    a = np.load(path)
    assert a.shape == (2, 128) and a.dtype == np.float16
    assert all(call["prompt"] == "" for call in encoder.model.calls)
    pq.write_table(pa.table({"row": [1, 0], "ctx_text": ["a", "b"]}), facts)
    with pytest.raises(ValueError, match="align"):
        context_vectors(facts, tmp_path / "other", encoder)
