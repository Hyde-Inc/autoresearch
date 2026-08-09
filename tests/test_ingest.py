from pathlib import Path

import pandas as pd

from autoresearch.ingest import ingest_csv, preview_csv


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
