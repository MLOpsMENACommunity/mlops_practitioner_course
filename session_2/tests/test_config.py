# tests/test_config.py
from src.config import DEFAULT_CONFIG_PATH, load_config


def test_default_config_path_is_relative():
    assert DEFAULT_CONFIG_PATH == "config/config.yaml"


def test_load_config_parses_yaml(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("data:\n  test_size: 0.2\n  seed: 42\n")

    cfg = load_config(cfg_file)

    assert cfg == {"data": {"test_size": 0.2, "seed": 42}}


def test_load_config_accepts_str_path(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("model:\n  n_estimators: 100\n")

    cfg = load_config(str(cfg_file))

    assert cfg["model"]["n_estimators"] == 100
