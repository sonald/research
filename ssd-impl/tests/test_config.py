from __future__ import annotations

from pathlib import Path

from ssd_impl.config import load_config


def test_config_extends_and_override(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "\n".join(
            [
                "project_root: .",
                "model:",
                "  name_or_path: base-model",
                "sample:",
                "  preset: post_ssd",
                "  presets:",
                "    post_ssd:",
                "      temperature: 1.1",
                "    frozen:",
                "      temperature: 0.7",
            ]
        ),
        encoding="utf-8",
    )
    child.write_text(
        "\n".join(
            [
                "extends: base.yaml",
                "model:",
                "  name_or_path: child-model",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(child, overrides=["training.max_steps=2", "sample.preset=frozen"])

    assert config.model.name_or_path == "child-model"
    assert config.training.max_steps == 2
    assert config.sample.preset == "frozen"
    assert config.project_root == tmp_path.resolve()
    assert config.selected_sample_decode().temperature == 0.7

