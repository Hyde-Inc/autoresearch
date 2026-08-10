import io
import json
import urllib.error
import urllib.request

import pandas as pd
import pyarrow as pa
import pytest

from autoresearch.foundry import FoundryError, parse_dataset_reference, read_dataset

RID = "ri.foundry.main.dataset.c26f11c8-cdb3-4f44-9f5d-9816ea1c82da"


def _arrow_bytes(frame: pd.DataFrame) -> bytes:
    table = pa.Table.from_pandas(frame)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_parse_dataset_reference_accepts_scheme_and_bare_rid() -> None:
    assert parse_dataset_reference(f"foundry://{RID}") == RID
    assert parse_dataset_reference(RID) == RID
    assert parse_dataset_reference(f"  foundry://{RID}  ") == RID


def test_parse_dataset_reference_ignores_local_paths() -> None:
    assert parse_dataset_reference("sales.csv") is None
    assert parse_dataset_reference("/tmp/data/sales.csv") is None
    assert parse_dataset_reference("ri.foundry.main.transaction.abc") is None


def test_parse_dataset_reference_rejects_malformed_scheme() -> None:
    with pytest.raises(FoundryError, match="not a dataset RID"):
        parse_dataset_reference("foundry://not-a-rid")


def test_read_dataset_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOUNDRY_HOSTNAME", raising=False)
    monkeypatch.delenv("FOUNDRY_TOKEN", raising=False)
    with pytest.raises(FoundryError, match="FOUNDRY_HOSTNAME and FOUNDRY_TOKEN"):
        read_dataset(RID)


def test_read_dataset_returns_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOUNDRY_HOSTNAME", "https://stack.palantirfoundry.com/")
    monkeypatch.setenv("FOUNDRY_TOKEN", "token-123")
    frame = pd.DataFrame({"date": ["2025-01-01"], "sku_name": ["a"], "sales": [3.0]})
    seen: dict[str, urllib.request.Request] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: int = 0):
        seen["request"] = request
        return _FakeResponse(_arrow_bytes(frame))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = read_dataset(RID, branch="master", row_limit=100)

    request = seen["request"]
    assert request.full_url.startswith(f"https://stack.palantirfoundry.com/api/v2/datasets/{RID}/readTable?")
    assert "format=ARROW" in request.full_url
    assert "branchName=master" in request.full_url
    assert "rowLimit=100" in request.full_url
    assert request.get_header("Authorization") == "Bearer token-123"
    pd.testing.assert_frame_equal(result, frame)


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, {}, "rejected the token"),
        (403, {}, "no permission"),
        (404, {}, "not found"),
        (400, {"errorName": "DatasetReadNotSupported"}, "cannot be read as a table"),
        (400, {"errorName": "ReadTableRowLimitExceeded"}, "row limit"),
        (500, {"message": "boom"}, "HTTP 500: boom"),
    ],
)
def test_read_dataset_maps_http_errors(
    monkeypatch: pytest.MonkeyPatch, status: int, body: dict, expected: str
) -> None:
    monkeypatch.setenv("FOUNDRY_HOSTNAME", "stack.palantirfoundry.com")
    monkeypatch.setenv("FOUNDRY_TOKEN", "token-123")

    def fake_urlopen(request, timeout=0):
        raise urllib.error.HTTPError(
            request.full_url, status, "error", {}, io.BytesIO(json.dumps(body).encode())
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(FoundryError, match=expected):
        read_dataset(RID)


def test_read_dataset_reports_unreachable_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOUNDRY_HOSTNAME", "stack.palantirfoundry.com")
    monkeypatch.setenv("FOUNDRY_TOKEN", "token-123")

    def fake_urlopen(request, timeout=0):
        raise urllib.error.URLError("nodename nor servname provided")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(FoundryError, match="could not reach stack.palantirfoundry.com"):
        read_dataset(RID)
