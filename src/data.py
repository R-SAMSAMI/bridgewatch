from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from src.constants import (
    DATA_DIR,
    FWF_FIELDS,
    FHWA_DOWNLOAD_URL,
    PROCESSED_DATA_PATH,
    PROCESSED_DIR,
    RAW_DIR,
    RAW_ZIP_PATH,
    REFERENCE_YEAR,
    STATE_CODE_TO_NAME,
)


def ensure_directories() -> None:
    for path in (DATA_DIR, RAW_DIR, PROCESSED_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_bridgewatch_data(force_refresh: bool = False) -> pd.DataFrame:
    ensure_directories()
    if PROCESSED_DATA_PATH.exists() and not force_refresh:
        try:
            return pd.read_csv(PROCESSED_DATA_PATH, low_memory=False)
        except Exception:
            pass

    raw_zip = _ensure_raw_zip(force_refresh=force_refresh)
    frame = _parse_fixed_width_zip(raw_zip)
    frame = _prepare_bridgewatch_dataset(frame)

    temp_output = PROCESSED_DIR / "bridgewatch_2025_processed.tmp.csv.gz"
    try:
        frame.to_csv(temp_output, index=False, compression="gzip")
        temp_output.replace(PROCESSED_DATA_PATH)
    except Exception:
        # If Windows is still holding the previous cache file open, keep serving
        # the rebuilt in-memory frame instead of failing the whole app.
        temp_output.unlink(missing_ok=True)
    return frame


def _ensure_raw_zip(*, force_refresh: bool) -> Path:
    if RAW_ZIP_PATH.exists() and not force_refresh:
        return RAW_ZIP_PATH

    with urlopen(FHWA_DOWNLOAD_URL, timeout=120) as response:
        RAW_ZIP_PATH.write_bytes(response.read())
    return RAW_ZIP_PATH


def _parse_fixed_width_zip(raw_zip: Path) -> pd.DataFrame:
    colspecs = [spec for _, spec in FWF_FIELDS]
    names = [name for name, _ in FWF_FIELDS]

    with zipfile.ZipFile(raw_zip) as archive:
        member = next(name for name in archive.namelist() if not name.endswith("/"))
        with archive.open(member) as file_handle:
            text_buffer = io.TextIOWrapper(file_handle, encoding="latin1")
            frame = pd.read_fwf(
                text_buffer,
                colspecs=colspecs,
                names=names,
                dtype=str,
            )
    return frame


def _clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().replace({"": np.nan, "N": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"": np.nan, "nan": np.nan})


def _prepare_bridgewatch_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()

    text_columns = [
        "state_code",
        "structure_number",
        "design_load",
        "type_of_service_on_bridge",
        "type_of_service_under_bridge",
        "kind_of_material_design",
        "type_of_design_construction",
        "scour_critical_bridges",
        "bridge_condition",
    ]
    numeric_columns = [
        "year_built",
        "lanes_on_structure",
        "average_daily_traffic",
        "skew",
        "number_of_spans_in_main_unit",
        "length_of_maximum_span",
        "structure_length",
        "bridge_roadway_width",
        "deck_width",
        "deck",
        "superstructure",
        "substructure",
        "operating_rating",
        "structural_evaluation",
        "designated_inspection_frequency",
        "year_reconstructed",
        "average_daily_truck_traffic",
        "lowest_condition_code",
        "deck_area",
    ]

    for column in text_columns:
        cleaned[column] = _clean_text(cleaned[column])
    for column in numeric_columns:
        cleaned[column] = _clean_numeric(cleaned[column])

    cleaned["state_code"] = cleaned["state_code"].str.zfill(2)
    cleaned["state_name"] = cleaned["state_code"].map(STATE_CODE_TO_NAME)
    cleaned = cleaned[cleaned["state_name"].notna()].copy()

    reconstructed = cleaned["year_reconstructed"].where(cleaned["year_reconstructed"] > 0)
    cleaned["reference_year"] = reconstructed.fillna(cleaned["year_built"])
    cleaned["bridge_age"] = REFERENCE_YEAR - cleaned["reference_year"]
    cleaned["bridge_age"] = cleaned["bridge_age"].clip(lower=0)

    cleaned["average_daily_truck_traffic"] = cleaned["average_daily_truck_traffic"].fillna(0)
    cleaned["truck_share"] = np.where(
        cleaned["average_daily_traffic"] > 0,
        cleaned["average_daily_truck_traffic"] / cleaned["average_daily_traffic"],
        0,
    )
    cleaned["truck_share"] = cleaned["truck_share"].clip(lower=0, upper=1)

    cleaned["bridge_condition"] = cleaned["bridge_condition"].fillna("Unknown")
    traffic_threshold = cleaned["average_daily_traffic"].quantile(0.75)
    cleaned["priority_review"] = (
        (cleaned["bridge_condition"] == "P")
        | (
            (cleaned["lowest_condition_code"] <= 5)
            & (cleaned["average_daily_traffic"] >= traffic_threshold)
        )
        | (cleaned["structural_evaluation"] <= 4)
    ).astype(int)

    cleaned["priority_label"] = cleaned["priority_review"].map(
        {1: "Priority Review", 0: "Routine Review"}
    )
    cleaned["condition_bucket"] = cleaned["bridge_condition"].map(
        {"G": "Good", "F": "Fair", "P": "Poor"}
    ).fillna("Unknown")

    cleaned = cleaned.dropna(
        subset=[
            "bridge_age",
            "average_daily_traffic",
            "lanes_on_structure",
            "number_of_spans_in_main_unit",
            "structure_length",
            "deck_width",
            "operating_rating",
            "deck",
            "superstructure",
            "substructure",
            "structural_evaluation",
        ]
    ).copy()

    cleaned["bridge_id"] = (
        cleaned["state_name"]
        + " | "
        + cleaned["structure_number"].fillna("Unknown")
    )
    return cleaned.reset_index(drop=True)
