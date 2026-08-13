"""Read Palantir Foundry datasets into DataFrames for ingestion.

Foundry exposes dataset contents through the public readTable endpoint
(``GET /api/v2/datasets/{rid}/readTable``), which streams the table as Arrow
IPC bytes. Credentials come from the environment (``FOUNDRY_HOSTNAME`` and
``FOUNDRY_TOKEN``, typically via ``.env``); the token needs the
``api:datasets-read`` scope and read access to the dataset.

The endpoint reads regular tabular datasets only - Views (virtual datasets
composed of other datasets) are unsupported, and non-Parquet-backed datasets
are capped at one million rows. Those failures are reported with actionable
messages instead of raw HTTP errors.
"""

from __future__ import annotations

import contextlib
import http.client
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATASET_RID_RE = re.compile(r"^ri\.foundry\.main\.dataset\.[0-9a-fA-F][0-9a-fA-F-]*$")
_SCHEME = "foundry://"


class FoundryError(RuntimeError):
    """A Foundry read failed in a way the user can act on."""


def parse_dataset_reference(source: str) -> str | None:
    """Return the dataset RID when ``source`` points at Foundry, else None.

    Accepts ``foundry://ri.foundry.main.dataset.<uuid>`` or a bare RID. A
    ``foundry://`` prefix with a malformed RID is an error rather than a
    silent fallthrough to the CSV path.
    """
    candidate = source.strip()
    if candidate.startswith(_SCHEME):
        rid = candidate[len(_SCHEME) :]
        if not DATASET_RID_RE.match(rid):
            raise FoundryError(
                f"'{rid}' is not a dataset RID; expected foundry://ri.foundry.main.dataset.<uuid>"
            )
        return rid
    if DATASET_RID_RE.match(candidate):
        return candidate
    return None


def _credentials() -> tuple[str, str]:
    hostname = os.getenv("FOUNDRY_HOSTNAME", "").strip()
    token = os.getenv("FOUNDRY_TOKEN", "").strip()
    missing = [
        name
        for name, value in (("FOUNDRY_HOSTNAME", hostname), ("FOUNDRY_TOKEN", token))
        if not value
    ]
    if missing:
        raise FoundryError(
            f"{' and '.join(missing)} not set; add them to .env "
            "(hostname like yourstack.palantirfoundry.com, token with api:datasets-read scope)"
        )
    hostname = hostname.removeprefix("https://").removeprefix("http://").rstrip("/")
    return hostname, token


def _describe_http_error(exc: urllib.error.HTTPError, rid: str) -> str:
    error_name = ""
    message = ""
    with contextlib.suppress(Exception):  # error bodies are best-effort
        payload = json.loads(exc.read().decode("utf-8", errors="replace"))
        error_name = str(payload.get("errorName") or payload.get("error", ""))
        message = str(payload.get("message") or payload.get("errorDescription", ""))

    if exc.code == 401:
        return "Foundry rejected the token (401); check FOUNDRY_TOKEN and that it has not expired"
    if exc.code == 403:
        return (
            f"the token has no permission to read dataset {rid} (403); "
            "grant the token's user read access or use a token with api:datasets-read scope"
        )
    if exc.code == 404:
        return f"dataset {rid} was not found (404); check the RID and the branch name"
    if "DatasetReadNotSupported" in error_name or "TypesNotSupported" in error_name:
        return (
            f"dataset {rid} cannot be read as a table ({error_name}); "
            "Views and some column types are not supported by readTable"
        )
    if "RowLimitExceeded" in error_name:
        return (
            f"dataset {rid} exceeds the readTable row limit ({error_name}); "
            "non-Parquet datasets are capped at 1M rows"
        )
    detail = message or error_name or exc.reason
    return f"Foundry readTable failed with HTTP {exc.code}: {detail}"


def read_dataset(
    rid: str,
    *,
    branch: str | None = None,
    columns: list[str] | None = None,
    row_limit: int | None = None,
    timeout_s: int = 300,
    retries: int = 3,
) -> pd.DataFrame:
    """Download a Foundry dataset as a DataFrame via the readTable endpoint.

    Transient network failures (dropped connections mid-stream, timeouts,
    5xx responses) are retried with a short backoff; 4xx responses fail fast.
    """
    hostname, token = _credentials()
    params: list[tuple[str, str]] = [("format", "ARROW")]
    if branch:
        params.append(("branchName", branch))
    for column in columns or []:
        params.append(("columns", column))
    if row_limit is not None:
        params.append(("rowLimit", str(row_limit)))
    url = (
        f"https://{hostname}/api/v2/datasets/{rid}/readTable?"
        f"{urllib.parse.urlencode(params)}"
    )

    payload: bytes | None = None
    last_error = "download failed"
    for attempt in range(retries):
        if attempt:
            time.sleep(2**attempt)  # 2s, 4s
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = response.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise FoundryError(_describe_http_error(exc, rid)) from exc
            last_error = _describe_http_error(exc, rid)
        except (
            urllib.error.URLError,
            http.client.IncompleteRead,
            http.client.HTTPException,
            TimeoutError,
            ConnectionError,
        ) as exc:
            reason = getattr(exc, "reason", None) or exc
            last_error = f"download from {hostname} was interrupted: {reason}"
    if payload is None:
        raise FoundryError(f"{last_error} (after {retries} attempts)")

    try:
        table = pa.ipc.open_stream(payload).read_all()
    except pa.ArrowInvalid as exc:
        raise FoundryError(f"Foundry returned data that is not an Arrow stream: {exc}") from exc
    return table.to_pandas()


# --------------------------------------------------------------------------- #
# Generic JSON API access (used by dataset-write and orchestration helpers).
# --------------------------------------------------------------------------- #


def _describe_api_error(exc: urllib.error.HTTPError, path: str) -> str:
    error_name = ""
    message = ""
    with contextlib.suppress(Exception):
        payload = json.loads(exc.read().decode("utf-8", errors="replace"))
        error_name = str(payload.get("errorName") or payload.get("error", ""))
        message = str(payload.get("message") or payload.get("errorDescription", ""))
    if exc.code == 401:
        return "Foundry rejected the token (401); check FOUNDRY_TOKEN and that it has not expired"
    if exc.code == 403:
        return (
            f"the token lacks permission for {path} (403); it needs the right scope "
            "(api:datasets-write for dataset writes, api:orchestration-write for builds)"
        )
    if exc.code == 404:
        return f"{path} was not found (404); check the RID / path"
    detail = message or error_name or exc.reason
    return f"Foundry API {path} failed with HTTP {exc.code}: {detail}"


def _api_json(
    method: str,
    path: str,
    *,
    body: dict | None = None,
    params: list[tuple[str, str]] | None = None,
    timeout_s: int = 120,
) -> dict:
    """Call a Foundry JSON endpoint and return the decoded response body."""
    hostname, token = _credentials()
    url = f"https://{hostname}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Authorization": f"Bearer {token}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise FoundryError(_describe_api_error(exc, path)) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", None) or exc
        raise FoundryError(f"Foundry request to {path} failed: {reason}") from exc
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return {}


# --------------------------------------------------------------------------- #
# Dataset branches.
# --------------------------------------------------------------------------- #


def get_branch(rid: str, branch: str) -> dict | None:
    """Return the branch record of a dataset, or None when it does not exist."""
    quoted = urllib.parse.quote(branch, safe="")
    try:
        return _api_json("GET", f"/api/v2/datasets/{rid}/branches/{quoted}")
    except FoundryError as exc:
        if "404" in str(exc):
            return None
        raise


def create_branch(rid: str, branch: str, *, transaction_rid: str | None = None) -> dict:
    """Create a dataset branch, optionally pointing at an existing transaction."""
    body: dict = {"name": branch}
    if transaction_rid:
        body["transactionRid"] = transaction_rid
    try:
        return _api_json("POST", f"/api/v2/datasets/{rid}/branches", body=body)
    except FoundryError as exc:
        if "400" not in str(exc):
            raise
        # Some stacks expect the v1-style field name.
        body = {"branchName": branch}
        if transaction_rid:
            body["transactionRid"] = transaction_rid
        return _api_json("POST", f"/api/v2/datasets/{rid}/branches", body=body)


# --------------------------------------------------------------------------- #
# Filesystem + dataset creation.
# --------------------------------------------------------------------------- #


def resolve_parent_folder(resource_rid: str) -> str:
    """Return the parent folder RID of any Foundry resource (e.g. the repo)."""
    payload = _api_json("GET", f"/api/v2/filesystem/resources/{resource_rid}")
    parent = payload.get("parentFolderRid") or payload.get("parentRid")
    if not parent:
        raise FoundryError(
            f"could not resolve the parent folder of {resource_rid}; "
            "pass the project folder RID explicitly"
        )
    return str(parent)


def find_child_dataset(folder_rid: str, name: str) -> str | None:
    """Return the RID of a dataset named ``name`` in ``folder_rid`` if present."""
    params: list[tuple[str, str]] = [("pageSize", "200")]
    next_token: str | None = None
    while True:
        page_params = list(params)
        if next_token:
            page_params.append(("pageToken", next_token))
        payload = _api_json(
            "GET", f"/api/v2/filesystem/folders/{folder_rid}/children", params=page_params
        )
        for child in payload.get("data", []) or payload.get("value", []):
            display = child.get("displayName") or child.get("name")
            rid = child.get("rid", "")
            if display == name and ".dataset." in rid:
                return rid
        next_token = payload.get("nextPageToken")
        if not next_token:
            return None


def find_child_folder(folder_rid: str, name: str) -> str | None:
    """Return the RID of a subfolder named ``name`` in ``folder_rid`` if present."""
    next_token: str | None = None
    while True:
        page_params: list[tuple[str, str]] = [("pageSize", "200")]
        if next_token:
            page_params.append(("pageToken", next_token))
        payload = _api_json(
            "GET", f"/api/v2/filesystem/folders/{folder_rid}/children", params=page_params
        )
        for child in payload.get("data", []) or payload.get("value", []):
            display = child.get("displayName") or child.get("name")
            rid = child.get("rid", "")
            if display == name and ".folder." in rid:
                return rid
        next_token = payload.get("nextPageToken")
        if not next_token:
            return None


def create_folder(name: str, parent_folder_rid: str) -> str:
    payload = _api_json(
        "POST",
        "/api/v2/filesystem/folders",
        body={"displayName": name, "parentFolderRid": parent_folder_rid},
    )
    rid = payload.get("rid")
    if not rid:
        raise FoundryError(f"folder create for '{name}' returned no RID: {payload}")
    return str(rid)


def ensure_folder(name: str, parent_folder_rid: str) -> str:
    existing = find_child_folder(parent_folder_rid, name)
    return existing or create_folder(name, parent_folder_rid)


def create_dataset(name: str, parent_folder_rid: str) -> str:
    """Create a dataset under ``parent_folder_rid`` and return its RID."""
    payload = _api_json(
        "POST",
        "/api/v2/datasets",
        body={"name": name, "parentFolderRid": parent_folder_rid},
    )
    rid = payload.get("rid")
    if not rid:
        raise FoundryError(f"dataset create for '{name}' returned no RID: {payload}")
    return str(rid)


def ensure_dataset(name: str, parent_folder_rid: str) -> str:
    """Return the RID of dataset ``name``, creating it if it does not exist."""
    existing = find_child_dataset(parent_folder_rid, name)
    return existing or create_dataset(name, parent_folder_rid)


# --------------------------------------------------------------------------- #
# Transactions, file upload, schema — the write path for a tabular dataset.
# --------------------------------------------------------------------------- #

_ARROW_TO_FOUNDRY = {
    "string": "STRING",
    "large_string": "STRING",
    "bool": "BOOLEAN",
    "int8": "INTEGER",
    "int16": "INTEGER",
    "int32": "INTEGER",
    "int64": "LONG",
    "float": "FLOAT",
    "float32": "FLOAT",
    "double": "DOUBLE",
    "date32[day]": "DATE",
    "date64[ms]": "DATE",
}


def _foundry_type(arrow_type: pa.DataType) -> str:
    key = str(arrow_type)
    if key in _ARROW_TO_FOUNDRY:
        return _ARROW_TO_FOUNDRY[key]
    if pa.types.is_timestamp(arrow_type):
        return "TIMESTAMP"
    if pa.types.is_integer(arrow_type):
        return "LONG"
    if pa.types.is_floating(arrow_type):
        return "DOUBLE"
    if pa.types.is_date(arrow_type):
        return "DATE"
    return "STRING"


def _field_schema_list(schema: pa.Schema) -> list[dict]:
    return [
        {"name": field.name, "type": _foundry_type(field.type), "nullable": True}
        for field in schema
    ]


def _create_transaction(rid: str, branch: str, transaction_type: str) -> str:
    payload = _api_json(
        "POST",
        f"/api/v2/datasets/{rid}/transactions",
        body={"transactionType": transaction_type},
        params=[("branchName", branch)],
    )
    tx = payload.get("rid")
    if not tx:
        raise FoundryError(f"could not open a transaction on {rid}: {payload}")
    return str(tx)


def _commit_transaction(rid: str, transaction_rid: str) -> None:
    _api_json("POST", f"/api/v2/datasets/{rid}/transactions/{transaction_rid}/commit")


def _abort_transaction(rid: str, transaction_rid: str) -> None:
    with contextlib.suppress(FoundryError):
        _api_json("POST", f"/api/v2/datasets/{rid}/transactions/{transaction_rid}/abort")


def _upload_bytes(
    rid: str, transaction_rid: str, file_path: str, data: bytes, timeout_s: int
) -> None:
    hostname, token = _credentials()
    quoted = urllib.parse.quote(file_path, safe="")
    params = urllib.parse.urlencode([("transactionRid", transaction_rid)])
    url = f"https://{hostname}/api/v2/datasets/{rid}/files/{quoted}/upload?{params}"
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        raise FoundryError(_describe_api_error(exc, f"upload to {rid}")) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", None) or exc
        raise FoundryError(f"upload to {rid} failed: {reason}") from exc


def upload_table(
    rid: str,
    frame: pd.DataFrame,
    *,
    branch: str = "master",
    timeout_s: int = 300,
) -> None:
    """Replace a dataset's contents with ``frame`` (SNAPSHOT) and apply its schema.

    Writes a single Parquet file inside one transaction, commits it, then calls
    putSchema so the dataset is immediately readable as a table.
    """
    table = pa.Table.from_pandas(frame, preserve_index=False)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    data = buffer.getvalue()

    transaction_rid = _create_transaction(rid, branch, "SNAPSHOT")
    try:
        _upload_bytes(rid, transaction_rid, "data.parquet", data, timeout_s)
        _commit_transaction(rid, transaction_rid)
    except FoundryError:
        _abort_transaction(rid, transaction_rid)
        raise

    _api_json(
        "PUT",
        f"/api/v2/datasets/{rid}/putSchema",
        body={
            "schema": {"fieldSchemaList": _field_schema_list(table.schema)},
            "dataframeReader": "PARQUET",
            "branchName": branch,
            "endTransactionRid": transaction_rid,
        },
    )
