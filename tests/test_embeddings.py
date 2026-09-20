import json

import numpy as np
import pandas as pd
import pytest

from jlink import block
from jlink.embeddings import serialize
from jlink.fields import parse_on


class Encoder:
    def __init__(self, lookup):
        self.lookup, self.calls = lookup, []

    def encode(self, texts, **kwargs):
        self.calls.append(texts)
        return np.array([self.lookup[t] for t in texts])


def test_semantic_pass_recovers_alias_and_preserves_lexical_sim_and_ids():
    a = pd.DataFrame({"id": ["NA"], "name": ["3M"]}, index=[91])
    b = pd.DataFrame({"id": ["001", "NULL"], "firm": ["Three Musketeers", "Minnesota Mining"]})
    encoder = Encoder({"3M": [1, 0], "Three Musketeers": [0, 1], "Minnesota Mining": [1, 0]})
    semantic = block.embeddings(("name", "firm"), k=1, encoder=encoder)
    result = block.candidates(a, b, on=[("name", "firm")], left_id="id", right_id="id",
                              blockers=[block.ngrams(("name", "firm"), k=1), semantic])
    assert list(zip(result.left_id, result.right_id)) == [("NA", "NULL")]
    assert result.sim.iloc[0] == 0  # semantic retrieval is not silently relabeled as lexical similarity
    assert result.attrs["blocking"]["passes"][1]["added_pairs"] == 1
    assert semantic.to_config()["custom_encoder"] and not semantic.to_config()["reconstructable"]
    json.dumps(semantic.to_config(), allow_nan=False)


def test_chunked_search_matches_dense_oracle_in_both_directions(monkeypatch):
    import jlink.embeddings as module

    monkeypatch.setattr(module, "_QUERY_ROWS", 2)
    monkeypatch.setattr(module, "_INDEX_ROWS", 3)
    rng = np.random.default_rng(421)
    x, y = rng.normal(size=(7, 5)), rng.normal(size=(11, 5))
    y[6] = y[0]  # boundary ties must prefer original row positions
    a, b = pd.DataFrame({"s": [f"a{i}" for i in range(7)]}), pd.DataFrame({"s": [f"b{i}" for i in range(11)]})
    encoder = Encoder(dict(zip([*a.s, *b.s], [*x, *y])))
    for reverse in (False, True):
        retriever = block.embeddings("s", k=4, min_sim=-1, encoder=encoder, reverse=reverse)
        xn, yn = retriever.vectors(a, b)
        product = yn @ xn.T if reverse else xn @ yn.T
        expected = []
        for i, scores in enumerate(product):
            for j in np.lexsort((np.arange(len(scores)), -scores))[:4]:
                expected.append((j, i) if reverse else (i, j))
        assert retriever.pairs(a, b).tolist() == [list(p) for p in expected]


def test_blank_and_zero_vectors_never_retrieve_even_at_negative_cutoff():
    a = pd.DataFrame({"s": [None, " ", "zero", "good", "good"]})
    b = pd.DataFrame({"s": ["good", "", "zero"]})
    enc = Encoder({"zero": [0, 0], "good": [1, 0]})
    retriever = block.embeddings("s", encoder=enc, min_sim=-1, k=3)
    assert retriever.pairs(a, b).tolist() == [[3, 0], [4, 0]]
    assert enc.calls == [["zero", "good"]]


def test_mapped_fields_preserve_boundaries_and_context():
    fields = parse_on([("first", "given"), ("last", "surname")])
    a = pd.DataFrame({"first": ["Mary Ann"], "last": ["Smith"]})
    b = pd.DataFrame({"given": ["Mary"], "surname": ["Ann Smith"]})
    assert serialize(a, fields, 1) != serialize(b, fields, 2)
    assert json.loads(serialize(b, fields, 2)[0]) == {"first": "Mary", "last": "Ann Smith"}


def test_within_maps_back_to_original_positions_and_limit_is_enforced():
    a = pd.DataFrame({"s": ["one", "two"], "state": ["NY", "CA"]})
    b = pd.DataFrame({"s": ["three", "four"], "state": ["CA", "NY"]})
    enc = Encoder({s: [1, 0] for s in [*a.s, *b.s]})
    retriever = block.embeddings("s", encoder=enc, k=2)
    assert block.within(retriever, "state").pairs(a, b).tolist() == [[0, 1], [1, 0]]
    with pytest.raises(ValueError, match="exceeding max_pairs"):
        block.candidates(a, b, on="s", blockers=[retriever], max_pairs=1)


@pytest.mark.parametrize("values", [[[float("nan"), 0]], [[float("inf"), 0]], [1], [[], []]])
def test_invalid_encoder_output(values):
    class Bad:
        def encode(self, *args, **kwargs):
            return values

    a = pd.DataFrame({"s": ["same"]})
    with pytest.raises(ValueError, match="finite 2D matrix"):
        block.embeddings("s", encoder=Bad()).pairs(a, a)


@pytest.mark.parametrize("kwargs", [{"k": True}, {"k": 0}, {"batch_size": 0}, {"min_sim": 2},
                                   {"min_sim": float("nan")}, {"reverse": 1}, {"model": ""},
                                   {"encoder": object()}, {"revision": ""}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        block.embeddings("name", **kwargs)


def test_empty_inputs_do_not_load_optional_model():
    retriever = block.embeddings("s", model="does-not-exist")
    assert retriever.pairs(pd.DataFrame({"s": []}), pd.DataFrame({"s": ["hello"]})).shape == (0, 2)
    assert retriever.pairs(pd.DataFrame({"s": [None]}), pd.DataFrame({"s": [""]})).shape == (0, 2)
