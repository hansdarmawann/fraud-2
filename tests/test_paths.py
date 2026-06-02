"""Tests for code/utils/paths.py"""

from pathlib import Path
import pytest


class TestPathConstants:
    def test_project_root_is_directory(self):
        from utils.paths import PROJECT_ROOT
        assert PROJECT_ROOT.is_dir()

    def test_all_output_dirs_exist(self):
        from utils.paths import DATA_RAW, DATA_PROCESSED, MODELS, REPORTS, LOGS
        for p in (DATA_RAW, DATA_PROCESSED, MODELS, REPORTS, LOGS):
            assert p.exists(), f"Expected directory to exist: {p}"

    def test_target_col_value(self):
        from utils.paths import TARGET_COL
        assert TARGET_COL == "is_fraud"

    def test_id_cols_value(self):
        from utils.paths import ID_COLS
        assert "transaction_id" in ID_COLS
        assert "user_id" in ID_COLS


class TestLoadConfig:
    def test_load_config_returns_dict(self):
        from utils.paths import load_config
        cfg = load_config()
        assert isinstance(cfg, dict)

    def test_config_has_required_keys(self):
        from utils.paths import load_config
        cfg = load_config()
        assert "preprocessing" in cfg
        assert "training" in cfg

    def test_preprocessing_values(self):
        from utils.paths import load_config
        cfg = load_config()
        pre = cfg["preprocessing"]
        assert pre["random_state"] == 42
        assert pre["test_size"] == 0.2

    def test_training_values(self):
        from utils.paths import load_config
        cfg = load_config()
        tr = cfg["training"]
        assert tr["time_budget_seconds"] == 120
        assert "lgbm" in tr["estimators"]
        assert tr["metric"] == "f1"
        assert tr["seed"] == 42
