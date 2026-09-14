"""Write an evidence-bound summary only after all comparison games complete."""

import argparse
import json
from pathlib import Path

from botcolosseo.cli.train_hierarchical_strategic import digest
from botcolosseo.evaluation.hierarchical_mixture_comparison import summarize_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing comparison summary")
    report = json.loads(args.report.read_text())
    result = summarize_comparison(report)
    result.update(report_sha256=digest(args.report), identity=report["identity"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(args.output)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
