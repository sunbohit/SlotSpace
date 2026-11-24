import json

import numpy as np
import pandas as pd
import pytest

from slotspace.artifacts import sha256
from slotspace.representations import (
    RepresentationIndex, build_representations, field_scores, representation_texts, tune_field_weights,
)


class Encoder:
    def metadata(self):
        return {"model": "fixture-256", "max_seq_length": 512}

    def documents(self, texts, dim, batch_size):
        return np.array([np.eye(dim)[0 if "red" in t else 1] for t in texts], dtype=np.float16)


def test_maxp_and_field_caches_are_topic_vectors_and_keep_relation_headers(tmp_path):
    docs = pd.DataFrame({"doc_id": [4, 17], "name": ["red condition", "blue drug"],
                         "doc_type": ["disease", "drug"], "text": ["red condition. More text.", "blue drug."],
                         "details_json": [json.dumps({"description": "red description"}), "{}"]})
    edges = pd.DataFrame({"src": [17, 4, 17], "dst": [4, 17, 4], "rel": ["indication"] * 3})
    fields = representation_texts(docs, "fields", edges)
    assert fields[fields.field == "rel:indication"].text.tolist() == ["indication: blue drug", "indication: red condition"]
    path, edge = tmp_path / "docs.parquet", tmp_path / "edges.parquet"
    docs.to_parquet(path, index=False)
    edges.to_parquet(edge, index=False)
    encoder = Encoder()
    build_representations(path, tmp_path / "maxp", encoder, "passages", batch_size=2)
    maxp = RepresentationIndex(tmp_path / "maxp", sha256(path), encoder.metadata())
    assert maxp.passage_scores(np.eye(256, dtype=np.float32)[0]).tolist() == [1, 0]
    with pytest.raises(ValueError, match="256"):
        maxp.passage_scores(np.eye(128, dtype=np.float32)[0])
    build_representations(path, tmp_path / "fields", encoder, "fields", edges=edge, batch_size=2)
    index = RepresentationIndex(tmp_path / "fields", sha256(path), encoder.metadata())
    matrix, work = index.field_features("red", np.eye(256, dtype=np.float32)[0], np.array([0, 1]))
    assert matrix.shape == (2, 3, 2) and work > 0


def test_field_coordinate_ascent_uses_validation_ranking_and_rejects_bad_weights():
    features = np.array([[[1, 0], [0, 0]], [[0, 0], [1, 0]]], dtype=float)
    records = [{"doc_ids": np.array([17, 91]), "features": features, "answers": [91]}]
    weights, trials = tune_field_weights(records, ["name", "details"])
    assert weights["name"] == 0 and weights["details"] > 0 and trials
    with pytest.raises(ValueError, match="inventory"):
        field_scores(features, {"name": 1}, ["name", "details"])
