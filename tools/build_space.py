"""Assemble the Hugging Face Space from this repository (one source of truth).

    python tools/build_space.py --output /path/to/new/folder

The folder contains exactly what to push to the Space: the interface, the EyeQC package, the frozen model, the
example photographs and the optional third-party tools.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_FILES = ("app.py", "engine.py", "ui.py", "research.py", "extras.py", "charts.py", "dashboard.py", "about.py", "style.css")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "generated", "interface_screenshot.png")


def build(output: Path) -> None:
    output = output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise SystemExit("Choose an output folder outside the repository.")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"{output} is not empty; choose a new folder.")
    output.mkdir(parents=True, exist_ok=True)
    for name in APP_FILES:
        shutil.copy2(ROOT / "app" / name, output / name)
    for source, target in ((ROOT / "app" / "examples", "examples"), (ROOT / "app" / "third_party", "third_party"), (ROOT / "eyeqc", "eyeqc"), (ROOT / "results", "results"), (ROOT / "assets", "assets")):
        shutil.copytree(source, output / target, ignore=IGNORE)
    (output / "models").mkdir()
    shutil.copy2(ROOT / "models" / "eyeqc_v3_model.json", output / "models" / "eyeqc_v3_model.json")
    shutil.copy2(ROOT / "app" / "space" / "README.md", output / "README.md")
    shutil.copy2(ROOT / "LICENSE", output / "LICENSE")
    requirements = (ROOT / "requirements.txt").read_text() + (ROOT / "requirements-app.txt").read_text()
    lines = [line for line in requirements.splitlines() if line and not line.startswith(("-r", "gradio", "torch"))]
    lines += ["torch==2.8.0", "torchvision==0.23.0"]            # versions supported by ZeroGPU; Gradio comes from sdk_version
    (output / "requirements.txt").write_text("\n".join(lines) + "\n")
    (output / ".gitattributes").write_text("*.jpg filter=lfs diff=lfs merge=lfs -text\n*.png filter=lfs diff=lfs merge=lfs -text\n")
    print(f"Space folder ready: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
