from pathlib import Path

import pandas as pd
import pytest

from autoresearch.ingest import apply_column_mapping, ingest_csv, ingest_frame, preview_csv


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
