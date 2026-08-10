from pathlib import Path

import pandas as pd
import pytest

from autoresearch.ingest import (
    apply_column_mapping,
    ingest_csv,
    ingest_frame,
    preview_csv,
    preview_frame,
)


def test_ingest_csv_creates_runnable_task(tmp_path: Path) -> None:
    dates = pd.date_range("2025-01-01", periods=15)
    frame = pd.DataFrame(
        [
            {
                "date": date,
                "sku_name": sku,
                "sales": 10 + index,
                "selling_price": 5.0,
            }
            for index, date in enumerate(dates)
            for sku in ("a", "b")
        ]
    )
    source = tmp_path / "sales.csv"
    frame.to_csv(source, index=False)

    preview = preview_csv(source)
    assert preview.ok
    assert preview.skus == 2

    result = ingest_csv(source, name="Demo Store", tasks_root=tmp_path / "tasks")
    assert result.config_path.exists()
    assert (result.task_dir / "seed/solution/train.py").exists()
    assert (result.task_dir / "private/holdout.parquet").exists()
    assert result.train_rows + result.validation_rows + result.holdout_rows == len(frame)


def test_ingest_frame_with_column_mapping_creates_task(tmp_path: Path) -> None:
    dates = pd.date_range("2025-01-01", periods=15)
    frame = pd.DataFrame(
        [
            {"day": date, "item_id": sku, "units_sold": 10 + index, "unit_price": 5.0}
            for index, date in enumerate(dates)
            for sku in ("a", "b")
        ]
    )
    mapped = apply_column_mapping(
        frame,
        id_column="item_id",
        date_column="day",
        target_column="units_sold",
        price_column="unit_price",
    )
    assert list(mapped.columns) == ["date", "sku_name", "sales", "selling_price"]

    result = ingest_frame(mapped, name="Foundry Demo", tasks_root=tmp_path / "tasks")
    assert result.config_path.exists()
    assert result.skus == 2
    assert result.train_rows + result.validation_rows + result.holdout_rows == len(frame)


def test_apply_column_mapping_rejects_unknown_columns() -> None:
    frame = pd.DataFrame({"date": [], "sku_name": [], "sales": [], "selling_price": []})
    with pytest.raises(ValueError, match="not found in the data: nope"):
        apply_column_mapping(frame, id_column="nope")


def test_preview_reports_per_column_quality_and_drops() -> None:
    dates = pd.date_range("2025-01-01", periods=10)
    rows = [
        {"date": date, "sku_name": sku, "sales": 10 + i, "selling_price": 5.0}
        for i, date in enumerate(dates)
        for sku in ("a", "b")
    ]
    # Inject unusable values: a bad date, a non-numeric price, and a blank sku.
    rows.append({"date": "not-a-date", "sku_name": "a", "sales": 1, "selling_price": 5.0})
    rows.append({"date": dates[0], "sku_name": "b", "sales": 2, "selling_price": "free"})
    rows.append({"date": dates[0], "sku_name": "  ", "sales": 3, "selling_price": 5.0})
    frame = pd.DataFrame(rows)

    preview = preview_frame(frame)

    assert preview.raw_rows == len(frame)
    assert preview.null_counts["date"] == 1
    assert preview.null_counts["selling_price"] == 1
    assert preview.null_counts["sku_name"] == 1
    assert preview.dropped_rows == 3
    assert 0 < preview.dropped_pct < 100
