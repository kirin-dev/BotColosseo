import hashlib
import json

import pytest

from botcolosseo.cli.run_deployment import load_bundle


def test_tampered_bundle_is_rejected_before_loading_models(tmp_path):
    asset = tmp_path / "scene.wad"
    asset.write_bytes(b"original")
    expected = hashlib.sha256(asset.read_bytes()).hexdigest()
    (tmp_path / "deployment.json").write_text(json.dumps({
        "schema": "botcolosseo-deployment-1", "files": {"scene.wad": expected}}))
    asset.write_bytes(b"modified")
    with pytest.raises(ValueError, match="integrity failure"):
        load_bundle(tmp_path)


def test_manifest_cannot_load_files_outside_bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    (tmp_path / "outside").write_bytes(b"outside")
    (root / "deployment.json").write_text(json.dumps({
        "schema": "botcolosseo-deployment-1", "files": {"../outside": "irrelevant"}}))
    with pytest.raises(ValueError, match="integrity failure"):
        load_bundle(root)
