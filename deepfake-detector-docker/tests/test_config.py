from pathlib import Path

import pytest

from deepfake_app.config import load_model_spec
from deepfake_app.errors import ResourceValidationError
from deepfake_app.resources import verify_assets


def test_manifest_matches_selected_interface_model() -> None:
    spec = load_model_spec()

    assert spec.checkpoint_path.name == "ft_spatial_spectral_bs64_lr1e-5_wd1e-3.pth"
    assert spec.base_frames == 60
    assert spec.num_frames == 50
    assert spec.num_dct == 256
    assert spec.threshold == pytest.approx(0.333)
    assert spec.backbone_weights == "IMAGENET1K_V1"


def test_packaged_assets_match_manifest_hashes() -> None:
    verify_assets(load_model_spec())


def test_manifest_rejects_missing_file(tmp_path: Path) -> None:
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text(
        '{"schema_version": 1, "checkpoint": {}}', encoding="utf-8"
    )

    with pytest.raises(ResourceValidationError):
        load_model_spec(manifest)
