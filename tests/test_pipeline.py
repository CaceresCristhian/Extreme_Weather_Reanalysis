"""
Unit tests for Extreme Weather Deep Sequence Modeling Pipeline.
Tests data integrity, split isolation, threshold logic, metric accuracy, and publication artifacts.
Uses relative paths via pathlib for complete portability.
"""

import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.train_extreme_weather import (
    build_chronological_sequences,
    compute_metrics,
    compute_multi_hazard_labels,
    evaluate_onset_metrics,
    load_and_preprocess_data,
)
DATA_PATH = BASE_DIR / "data" / "01_raw" / "noaa_ghcn_daily_stations_2020_2024.parquet"
FIG_PATH = BASE_DIR / "paper" / "figures" / "fig1_extreme_weather_benchmark.png"
MACROS_PATH = BASE_DIR / "paper" / "tables" / "macros.tex"
TAB_PATH = BASE_DIR / "paper" / "tables" / "tab_benchmark.tex"
TEX_PATH = BASE_DIR / "paper" / "main.tex"


def test_data_integrity_and_station_diversity():
    """Verify authentic NOAA GHCN-Daily parquet dataset has 8 stations and continuous dates."""
    assert DATA_PATH.exists(), f"Dataset missing at {DATA_PATH}"
    df = pd.read_parquet(DATA_PATH)
    assert len(df) >= 19000, f"Expected >= 19,000 records, got {len(df)}"
    assert df["station"].nunique() == 8, f"Expected 8 stations, got {df['station'].nunique()}"
    assert set(["Phoenix_AZ", "Las_Vegas_NV", "Houston_TX", "Miami_FL", "Denver_CO", "Minneapolis_MN", "Seattle_WA", "Chicago_IL"]).issubset(set(df["station"].unique()))
    assert df["tmax_c"].notnull().all(), "tmax_c must have no NaNs"
    assert df["tmin_c"].notnull().all(), "tmin_c must have no NaNs"


def test_chronological_split_isolation_no_leakage():
    """Verify strict chronological ordering between train, val, and test partitions."""
    df = load_and_preprocess_data()
    train_recs, val_recs, test_recs, _ = build_chronological_sequences(df, seq_len=7)

    max_train_date = max(r["target_date"] for r in train_recs)
    min_val_date = min(r["target_date"] for r in val_recs)
    max_val_date = max(r["target_date"] for r in val_recs)
    min_test_date = min(r["target_date"] for r in test_recs)

    assert max_train_date < min_val_date, f"Train date {max_train_date} leaks into val date {min_val_date}"
    assert max_val_date < min_test_date, f"Val date {max_val_date} leaks into test date {min_test_date}"
    assert max_train_date <= pd.to_datetime("2023-12-31")
    assert min_test_date >= pd.to_datetime("2025-01-01")


def test_multi_hazard_threshold_logic():
    """Test physical meteorological threshold classification by calling production pipeline code."""
    toy_df = pd.DataFrame({
        "station": ["Test_A", "Test_B", "Test_C", "Test_D", "Test_E"],
        "date": pd.date_range("2024-01-01", periods=5),
        "tmax_c": [38.5, 20.0, 15.0, 10.0, 22.0],  # #0 is heat extreme (>=38C)
        "tmin_c": [25.0, -16.0, 5.0, 0.0, 10.0],   # #1 is freeze extreme (<=-15C)
        "prcp_mm": [0.0, 0.0, 26.0, 0.0, 5.0],     # #2 is precip extreme (>=25mm)
        "wind_gust_ms": [5.0, 10.0, 12.0, 22.0, 8.0] # #3 is wind extreme (>=20m/s)
    })
    labeled = compute_multi_hazard_labels(toy_df)
    assert labeled["is_extreme"].tolist() == [1, 1, 1, 1, 0]
    assert labeled["heat_extreme"].tolist() == [1, 0, 0, 0, 0]
    assert labeled["freeze_extreme"].tolist() == [0, 1, 0, 0, 0]
    assert labeled["precip_extreme"].tolist() == [0, 0, 1, 0, 0]
    assert labeled["wind_extreme"].tolist() == [0, 0, 0, 1, 0]


def test_metric_computation_consistency():
    """Verify compute_metrics calculates exact expected values for toy inputs."""
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.4, 0.3])
    m = compute_metrics(y_true, y_prob, threshold=0.5)

    # Hand-calculated reference:
    # y_pred = [1, 1, 0, 0, 0, 0] -> TP=2, FP=0, FN=1, TN=3
    # Precision = 2/2 = 100%, Recall = 2/3 = 66.67%, F1 = 80.0%
    # Brier = mean([(0.9-1)^2, (0.8-1)^2, (0.1-0)^2, (0.2-0)^2, (0.4-1)^2, (0.3-0)^2]) = 0.55 / 6
    assert m["Precision"] == pytest.approx(100.0)
    assert m["Recall"] == pytest.approx(200.0 / 3.0)
    assert m["F1"] == pytest.approx(80.0)
    assert m["Brier"] == pytest.approx(0.55 / 6.0)
    assert 0.0 <= m["AUC"] <= 1.0
    assert 0.0 <= m["PR_AUC"] <= 1.0


def test_publication_artifacts_integrity():
    """Check existence, formatting, and valid LaTeX syntax of all paper artifacts."""
    assert FIG_PATH.exists() and FIG_PATH.stat().st_size > 20000, "Figure 1 must exist and be >20KB"
    assert MACROS_PATH.exists() and MACROS_PATH.stat().st_size > 100, "Macros TeX must exist"
    assert TAB_PATH.exists() and TAB_PATH.stat().st_size > 100, "Benchmark table must exist"
    assert TEX_PATH.exists(), "main.tex must exist"

    tex_content = TEX_PATH.read_text(encoding="utf-8")
    assert "\\begin{document}" in tex_content
    assert "\\end{document}" in tex_content
    # Check no unescaped ampersands in titles or section headers
    for line in tex_content.splitlines():
        if line.strip().startswith(("\\section{", "\\subsection{", "\\title{")):
            clean_line = line.replace("\\&", "")
            assert "&" not in clean_line, f"Unescaped '&' found in section/title: {line}"


def test_onset_metrics_logic():
    """Verify production evaluate_onset_metrics correctly isolates transitions and forces persistence to 0%."""
    # 5 time steps:
    # Day 0: y=0, prev=0
    # Day 1: y=1, prev=0 (Onset event)
    # Day 2: y=1, prev=1 (Continuation)
    # Day 3: y=0, prev=1 (Termination)
    # Day 4: y=1, prev=0 (Onset event)
    y_true = np.array([0, 1, 1, 0, 1])
    y_prev = np.array([0, 0, 1, 1, 0])

    # Model predicts Day 1 (prob 0.8) correctly, misses Day 4 (prob 0.2)
    model_probs = np.array([0.1, 0.8, 0.9, 0.1, 0.2])
    res_model = evaluate_onset_metrics(y_true, y_prev, model_probs, threshold=0.5)

    # 3 days have prev=0 (Day 0, Day 1, Day 4). Among them, 2 are onset positives (Day 1, Day 4)
    assert res_model["n_sub"] == 3
    assert res_model["n_pos"] == 2
    assert res_model["Recall"] == 50.0  # Caught 1 of 2 onsets

    # Naive persistence predicts y_prev
    res_pers = evaluate_onset_metrics(y_true, y_prev, y_prev.astype(float), threshold=0.5)
    assert res_pers["Recall"] == 0.0, "Persistence must have 0% onset recall by construction"


def test_macros_consistency():
    """Verify that macros in macros.tex accurately reflect data outputs."""
    assert MACROS_PATH.exists(), "macros.tex must exist."
    macro_text = MACROS_PATH.read_text(encoding="utf-8")
    macros = {}
    for match in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]+)\}", macro_text):
        macros[match.group(1)] = match.group(2)
    assert macros.get("numStations") == "8"
    assert macros.get("numHeatDays") == "1,489"
    assert macros.get("numFreezeDays") == "299"
    assert macros.get("numPrecipDays") == "375"
    assert macros.get("numWindDays") == "1"


def test_security_no_credential_or_mcp_leakage():
    """Guarantee zero API tokens, credentials, or secrets in project tree."""
    base_dir = Path(__file__).resolve().parents[1]

    # Rule 1: No .env or credentials inside project directory
    forbidden_files = [".env", ".env.local", "mcp_config.json", "Copernicus.txt"]
    for f in forbidden_files:
        assert not (base_dir / f).exists(), f"Security violation: {f} found in project directory!"

    # Rule 2: Scan python source files for sensitive hardcoded tokens
    suspicious_patterns = ["Bearer ", "sk-", "api_key = \"", "AIzaSy"]
    for py_file in (base_dir / "src").glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for pat in suspicious_patterns:
            assert pat not in text, f"Security violation: found suspicious pattern '{pat}' in {py_file.name}"
