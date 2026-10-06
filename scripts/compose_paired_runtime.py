"""Compose the audited Defensive case without modifying either policy rollout."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clips = [args.directory / f"defensive-{arm}.mp4" for arm in ("control", "switch")]
    reports = [json.loads(p.with_suffix(".json").read_text()) for p in clips]
    left, right = reports
    assert left["observation_hashes"][:82] == right["observation_hashes"][:82]
    assert [(r["action"], r["command"]) for r in left["report"]["timeline"][:81]] == [
        (r["action"], r["command"]) for r in right["report"]["timeline"][:81]]
    for key in ("executor", "strategy", "opponent", "scenario", "seed", "learner_role"):
        assert left[key] == right[key]
    assert left["seed"] == 104 and left["learner_role"] == "host"
    assert left["report"]["payoffs"][0] == right["report"]["payoffs"][0] == 0.3
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    def text(value, x, y, size, color, extra=""):
        return (f"drawtext=fontfile={font}:text='{value}':x={x}:y={y}:"
                f"fontsize={size}:fontcolor={color}{extra}")
    clean = "drawbox=x=0:y=27:w=490:h=24:color=0x0e0e0e:t=fill"
    left_filter = clean + "," + text("NEUTRAL | FIRST-PERSON POLICY", 10, 33, 12, "0x50b5cc")
    right_filter = clean + "," + text(
        "NEUTRAL | FIRST-PERSON POLICY", 10, 33, 12, "0x50b5cc", ":enable='lt(t,10.057143)'")
    right_filter += "," + text(
        "DEFENSIVE | FIRST-PERSON POLICY", 10, 33, 12, "0x50b5cc",
        ":enable='gte(t,10.057143)'")
    captions = [text("KEEP NEUTRAL", 20, 14, 23, "0x243047"),
                text("SWITCH TO DEFENSIVE", 660, 14, 23, "0x23816b"),
                text("Same policy and difficulty | Shared history at the switch", 20, 43, 14,
                     "0x526075"),
                text("Control requested at 9.26 s", 660, 43, 14, "0x23816b"),
                text("EXTRACTED 45 | 32.80 s", 790, 250, 25, "white",
                     ":box=1:boxcolor=0x23816b@0.9:boxborderw=12:enable='gte(t,32.8)'")]
    filters = (f"[0:v]{left_filter}[left];[1:v]{right_filter},"
               "tpad=stop_mode=clone:stop_duration=7[right];"
               "[left][right]hstack=inputs=2:shortest=1,pad=1280:520:0:64:white,"
               + ",".join(captions) + "[out]")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(clips[0]),
                    "-i", str(clips[1]), "-filter_complex", filters, "-map", "[out]", "-an",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "21", "-pix_fmt", "yuv420p",
                    str(args.output)], check=True)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", "12",
                    "-i", str(args.output), "-frames:v", "1", str(args.output.with_suffix(".jpg"))],
                   check=True)


if __name__ == "__main__":
    main()
