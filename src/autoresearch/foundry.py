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
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd
import pyarrow as pa

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
