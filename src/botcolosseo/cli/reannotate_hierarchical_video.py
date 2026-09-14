"""Reflow recorded event captions without replaying or changing game events."""

import argparse
import json
from pathlib import Path

import cv2

from botcolosseo.cli.train_hierarchical_strategic import digest
from botcolosseo.envs.video import write_mp4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence_path = args.output.with_suffix(".json")
    if args.output.exists() or evidence_path.exists():
        raise FileExistsError("Preserve original and previously derived videos")
    report = json.loads(args.evidence.read_text())
    if digest(args.video) != report["video"] or report["fps"] != 35:
        raise ValueError("Source video does not match its 35fps event evidence")
    events = report["events"]
    if [e["seconds"] for e in events] != sorted(e["seconds"] for e in events):
        raise ValueError("Event timeline must be ordered")
    capture = cv2.VideoCapture(str(args.video))

    def frames():
        count, event_index = 0, -1
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if frame.shape != (440, 640, 3):
                raise ValueError("Reannotation expects the recorded hierarchical layout")
            # Each recorded low-step frame was repeated for its four engine tics.
            seconds = (count // 4 + 1) * 4 / 35
            while (
                event_index + 1 < len(events)
                and events[event_index + 1]["seconds"] <= seconds + 1e-8
            ):
                event_index += 1
            frame[326:366] = 14
            if event_index >= 0 and seconds - events[event_index]["seconds"] < 18 * 4 / 35 - 1e-8:
                lines = [""]
                for word in events[event_index]["text"].split():
                    trial = (lines[-1] + " " + word).strip()
                    if cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0] > 620:
                        lines.append(word)
                    else:
                        lines[-1] = trial
                if len(lines) > 3:
                    raise ValueError("Caption exceeds reserved strip")
                for index, line in enumerate(lines):
                    cv2.putText(
                        frame,
                        line,
                        (10, 338 + 12 * index),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.36,
                        (80, 220, 255),
                        1,
                        cv2.LINE_AA,
                    )
            count += 1
            yield frame
        if count != report["frames"]:
            raise ValueError("Decoded frame count differs from original evidence")

    try:
        write_mp4(frames(), args.output, fps=35)
    finally:
        capture.release()
    report["derivation"] = {
        "source_video": str(args.video),
        "source_video_sha256": report["video"],
        "source_evidence_sha256": digest(args.evidence),
        "code_sha256": digest(Path(__file__)),
        "operation": (
            "reflow event caption strip only; unchanged sequence and event data; reencoded"
        ),
    }
    report["video"] = digest(args.output)
    temporary = evidence_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(evidence_path)
    print(json.dumps(report["derivation"]))


if __name__ == "__main__":
    main()
