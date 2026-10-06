import hashlib
import json

import pytest

from botcolosseo.cli.run_deployment import load_bundle


def test_archive_keeps_dotted_release_version():
    import importlib.util
    from pathlib import Path

    path = Path("scripts/build_deployment_bundle.py")
    spec = importlib.util.spec_from_file_location("bundle_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.archive_path(Path("bundle-v0.1.1")) == Path("bundle-v0.1.1.tar.gz")


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
