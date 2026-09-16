"""serialize: every result dataclass survives a JSON round trip; properties ride along."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from basic_kb.models import (
    FileError, IndexResult, InstanceInfo, PreviewChunk, PreviewFile, ReindexResult, ScanResult, SearchResult,
    SourceInfo, SourceStatus, VacuumResult,
)
from basic_kb.serialize import from_dict, to_jsonable


def roundtrip(obj):
    data = json.loads(json.dumps(to_jsonable(obj)))
    return data, from_dict(type(obj), data)


def test_search_result_round_trip():
    r = SearchResult(doc="d", metadata={"title": "T", "n": 1}, score=0.5, rerank_score=None)
    data, back = roundtrip(r)
    assert data["sort_key"] == 0.5             # property included
    assert back == r


def test_index_result_with_nested_errors():
    r = IndexResult(source_id="s", label="L", files_on_disk=3, added=1, errors=[FileError("a.md", "boom")],
                    limited_to=None, abort_reason=None)
    data, back = roundtrip(r)
    assert data["embedded"] == 1 and data["errors"] == [{"rel_path": "a.md", "error": "boom"}]
    assert back == r and isinstance(back.errors[0], FileError)


def test_source_status_round_trip():
    st = SourceStatus(source_id="s", label="L", directory="/d", store_dir="/s", model_id="m",
                      directory_exists=True, indexed=True, chunks=10, chars=400, new=1, updated=2,
                      deleted=3, indexed_at=1.5, content_types={"a": 4})
    data, back = roundtrip(st)
    assert (data["stale"], data["pending"], data["approx_tokens"]) == (6, 3, 100)
    assert back == st


def test_instance_info_nested_list():
    inf = InstanceInfo(name="n", model_id="m", store_dir="/s",
                       sources=[SourceInfo("a", "A", "", "markdown", "breadcrumb", 2, 3, True)])
    data, back = roundtrip(inf)
    assert data["total_chunks"] == 3 and data["sources"][0]["chunks_per_file"] == 1.5
    assert back == inf and isinstance(back.sources[0], SourceInfo)


def test_scan_and_reindex_results():
    s = ScanResult("s", "L", True, 4, 1, 1, 1, 1)
    r = ReindexResult("s", embedded=2, chunks_embedded=5)
    assert roundtrip(s)[1] == s and roundtrip(s)[0]["stale"] == 3
    assert roundtrip(r)[1] == r and roundtrip(r)[0]["changed"] is True


def test_preview_and_vacuum_results():
    pf = PreviewFile("a.md", 120, [PreviewChunk("text", "# A"), PreviewChunk("more")])
    data, back = roundtrip(pf)
    assert data["skipped"] is False and back == pf and isinstance(back.chunks[1], PreviewChunk)
    vr = VacuumResult(True, "/s/kb.sqlite3", 4096, 8, 0)
    assert roundtrip(vr)[1] == vr


def test_paths_become_strings():
    assert to_jsonable({"p": Path("/x/y"), "l": (Path("a"),)}) == {"p": "/x/y", "l": ["a"]}


def test_from_dict_reports_missing_required_field():
    with pytest.raises(ValueError, match="missing field 'label'"):
        from_dict(ScanResult, {"source_id": "s"})


def test_from_dict_ignores_unknown_keys():
    back = from_dict(ReindexResult, {"source_id": "s", "embedded": 1, "changed": True, "bogus": 9})
    assert back == ReindexResult("s", embedded=1)
