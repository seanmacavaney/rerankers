"""Tests for the PyTerrier integration (:mod:`rerankers.integrations.pyterrier`).

These tests use a fake reranker so they run without downloading a model. They are skipped entirely when
PyTerrier is not installed (e.g., when ``rerankers[pyterrier]`` has not been installed).
"""

import pandas as pd
import pytest

# The PyTerrier integration imports ``pyterrier`` at module load, so skip everything if it is missing.
pt = pytest.importorskip("pyterrier")

from rerankers.documents import Document
from rerankers.integrations.pyterrier import RerankersTransformer
from rerankers.models.ranker import BaseRanker
from rerankers.results import RankedResults, Result


class FakeRanker(BaseRanker):
    """A minimal reranker that scores documents by their input position.

    The last document passed to :meth:`rank` receives the highest score, so reranking deterministically
    reverses the input order. This lets us test the transformer without loading a real model.
    """

    def __init__(self):
        pass

    def score(self, query, doc):
        return 0.0

    def rank(self, query, docs, doc_ids=None):
        results = [
            Result(
                document=Document(doc_id=doc_id, text=text),
                score=float(position),
                rank=position,
            )
            for position, (text, doc_id) in enumerate(zip(docs, doc_ids))
        ]
        ranked = sorted(results, key=lambda r: r.score, reverse=True)
        return RankedResults(results=ranked, query=query, has_scores=True)


def _result_frame(rows):
    return pd.DataFrame(rows, columns=["qid", "query", "docno", "text", "score", "rank"])


def test_reranks_single_query():
    transformer = RerankersTransformer(FakeRanker())
    inp = _result_frame([
        ["q1", "hello", "d0", "first doc", 9.0, 0],
        ["q1", "hello", "d1", "second doc", 8.0, 1],
        ["q1", "hello", "d2", "third doc", 7.0, 2],
    ])

    out = transformer.transform(inp)

    # FakeRanker scores by input position, so the order is reversed and ranks are reassigned.
    assert list(out["docno"]) == ["d2", "d1", "d0"]
    assert list(out["score"]) == [2.0, 1.0, 0.0]
    assert list(out["rank"]) == [0, 1, 2]
    # Existing columns (e.g., the query and text) are preserved.
    assert set(out["query"]) == {"hello"}
    assert out.loc[out["docno"] == "d0", "text"].item() == "first doc"


def test_reranks_each_query_independently():
    transformer = RerankersTransformer(FakeRanker())
    inp = _result_frame([
        ["q1", "hello", "d0", "a", 9.0, 0],
        ["q1", "hello", "d1", "b", 8.0, 1],
        ["q2", "world", "d2", "c", 5.0, 0],
        ["q2", "world", "d3", "d", 4.0, 1],
        ["q2", "world", "d4", "e", 3.0, 2],
    ])

    out = transformer.transform(inp)

    q1 = out[out["qid"] == "q1"]
    q2 = out[out["qid"] == "q2"]

    # Each query is reranked and ranked independently, with ranks restarting from 0.
    assert list(q1["docno"]) == ["d1", "d0"]
    assert list(q1["rank"]) == [0, 1]
    assert list(q2["docno"]) == ["d4", "d3", "d2"]
    assert list(q2["rank"]) == [0, 1, 2]


def test_handles_empty_frame():
    transformer = RerankersTransformer(FakeRanker())
    out = transformer.transform(_result_frame([]))
    assert len(out) == 0
    # The score/rank columns are still emitted so the output schema (and schematics) stay correct.
    assert "score" in out.columns
    assert "rank" in out.columns


def test_missing_required_column_is_rejected():
    transformer = RerankersTransformer(FakeRanker())
    # No ``text`` column -> not a valid input for the reranker.
    bad = pd.DataFrame([{"qid": "q1", "query": "hello", "docno": "d0"}])
    with pytest.raises(pt.validate.InputValidationError):
        transformer.transform(bad)


def test_schematic_label():
    from pyterrier.schematic import HasSchematic

    # FakeRanker exposes no model name, so the label falls back to the ranker's class name.
    transformer = RerankersTransformer(FakeRanker())
    assert isinstance(transformer, HasSchematic)
    assert transformer.schematic(input_columns=None) == {"label": "FakeRanker"}
