"""Build a minimal inference artifact, preserving original checkpoints."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", help="Published source commit for this bundle")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".tar.gz").exists():
        raise FileExistsError("Preserve previous bundle")
    source = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True)
    # Export only inference source and the exact scene; no Git history or private notes.
    for directory in ("src", "assets/scenarios/crystal_run_extraction_randomized", "licenses"):
        shutil.copytree(source / directory, args.output / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(source / name, args.output / name)
    paths = {
        "executor": "runs/hierarchical/difficulty-anchored-pilot/epoch-3.pt",
        "strategy": "runs/hierarchical/random-switch-difficulty10k/last.pt",
        "opponent": "runs/hierarchical/difficulty-anchored-pilot/control-bundle/strategy-1.pt",
    }
    expected = json.loads((source / "docs/assets/hierarchical/task-difficulty.json").read_text())[
        "identities"]
    originals = {}
    (args.output / "models").mkdir()
    for name, relative in paths.items():
        original = args.artifact_root / relative
        originals[name] = digest(original)
        if originals[name] != expected[name]:
            raise ValueError(f"Selected model identity mismatch: {name}")
        payload = torch.load(original, map_location="cpu", weights_only=False)
        if name != "executor" and payload["identity"]["executor"] != expected["executor"]:
            raise ValueError("High-level model uses a different executor")
        target = args.output / "models" / f"{name}.pt"
        torch.save(payload["actor"], target)
        restored = torch.load(target, map_location="cpu", weights_only=True)
        if restored.keys() != payload["actor"].keys() or not all(
            torch.equal(value, restored[key]) for key, value in payload["actor"].items()
        ):
            raise ValueError("Export changed model parameters")
    scene = "assets/scenarios/crystal_run_extraction_randomized"
    scenario_hash = json.loads((args.output / scene / "manifest.json").read_text())["wad_sha256"]
    if digest(args.output / scene / "crystal_run_extraction_randomized.wad") != scenario_hash:
        raise ValueError("Scene WAD differs from its manifest")
    packages = ("torch", "vizdoom", "gymnasium", "numpy", "PyYAML", "imageio",
                "imageio-ffmpeg", "opencv-python-headless")
    versions = {name: importlib.metadata.version(name) for name in packages}
    (args.output / "requirements-runtime.txt").write_text("\n".join(
        f"{name}=={version.split('+')[0]}" for name, version in versions.items()) + "\n")
    shutil.copy2(source / "docs/deployment-bundle.md", args.output / "README.md")
    manifest = {"schema": "botcolosseo-deployment-1", "showcase_commit":
                "0c792ea91dd5755d9408bf8ee1049bb0378d8392",
                "source_commit": args.source_commit or subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=source).decode().strip(),
                "source_checkpoint_sha256": originals, "environment_versions": versions,
                "scenario_hash": scenario_hash,
                "scenario_config": scene + "/crystal_run_extraction_randomized.cfg",
                "files": {str(p.relative_to(args.output)): digest(p)
                          for p in sorted(args.output.rglob("*")) if p.is_file()}}
    (args.output / "deployment.json").write_text(json.dumps(manifest, indent=2))
    archive = args.output.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(args.output, arcname=args.output.name)
    print(json.dumps({"bundle": str(args.output), "archive": str(archive),
                      "bytes": archive.stat().st_size, "sha256": digest(archive)}))


if __name__ == "__main__":
    main()
