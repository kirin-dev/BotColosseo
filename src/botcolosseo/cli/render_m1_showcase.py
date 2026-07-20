from pathlib import Path

from botcolosseo.demo.m1_showcase import render_showcase


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    for path in render_showcase(root):
        print(path)
    return 0
