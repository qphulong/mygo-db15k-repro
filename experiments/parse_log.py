"""Parse MyGO training logs into per-epoch loss/time series and per-eval
Link-Prediction metric rows, and summarize the best epochs by validation
and by test MRR.

Standard library only (so this module can be inlined verbatim into the
Kaggle notebook).

Expected log format (from the MyGO author's training script, logging
format ``'%(asctime)s - %(levelname)s - %(message)s'``):

    2026-09-19 10:00:01,123 - INFO - 1 \t 812.345678 \t 9.876543 s

    2026-09-19 10:08:13,827 - INFO - Link Prediction on Validation Set
    2026-09-19 10:08:13,828 - INFO - MR: 145.7
    2026-09-19 10:08:13,829 - INFO - MRR: 0.301234
    2026-09-19 10:08:13,830 - INFO - Hit10: 0.5523
    2026-09-19 10:08:13,831 - INFO - Hit3: 0.3821
    2026-09-19 10:08:13,832 - INFO - Hit1: 0.2015
    2026-09-19 10:08:14,000 - INFO - Link Prediction on Test Set
    ... (same five metric lines) ...

Other lines (args dump, tqdm progress bars, tensor dumps, etc.) are
ignored.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

# Per-epoch training line: "<epoch> \t <loss> \t <cumulative_seconds> s"
# anchored on "- <digits>" so it does not match anywhere else in the line.
EPOCH_LINE_RE = re.compile(r"-\s(\d+)\s*\t\s*([\d.]+)\s*\t\s*([\d.]+) s")

# Eval-block headers.
VALID_HEADER_RE = re.compile(r"Link Prediction on Validation Set")
TEST_HEADER_RE = re.compile(r"Link Prediction on Test Set")

# Metric line: key is matched exactly via alternation + trailing ": " so
# "MR: " is never confused with "MRR: ".
METRIC_LINE_RE = re.compile(r"\b(MR|MRR|Hit10|Hit3|Hit1): ([\d.]+)")


def parse_log(path: str) -> list[dict]:
    """Parse a MyGO log file into a list of per-eval rows.

    Each row is a dict with keys: epoch (int), split ('valid'|'test'),
    MR, MRR, Hit10, Hit3, Hit1 (float). Rows are emitted in the order
    they appear in the log (valid block first, then test block, per
    eval point).
    """
    rows: list[dict] = []
    last_epoch = None
    current_split = None
    current_epoch = None
    current_metrics: dict | None = None

    def flush():
        if current_split is not None and current_metrics is not None and current_epoch is not None:
            row = {"epoch": current_epoch, "split": current_split}
            row.update(current_metrics)
            rows.append(row)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            epoch_match = EPOCH_LINE_RE.search(line)
            if epoch_match:
                last_epoch = int(epoch_match.group(1))
                continue

            if VALID_HEADER_RE.search(line):
                flush()
                current_split = "valid"
                current_epoch = last_epoch
                current_metrics = {}
                continue

            if TEST_HEADER_RE.search(line):
                flush()
                current_split = "test"
                current_epoch = last_epoch
                current_metrics = {}
                continue

            if current_metrics is not None:
                metric_match = METRIC_LINE_RE.search(line)
                if metric_match:
                    key, value = metric_match.group(1), metric_match.group(2)
                    current_metrics[key] = float(value)

    flush()
    return rows


def summarize(rows: list[dict]) -> dict:
    """Summarize eval rows into best-by-valid and best-by-test.

    best_by_valid: the epoch with the highest validation MRR, reported
    using that epoch's TEST-set metrics.
    best_by_test: the epoch with the highest test MRR, reported using
    that epoch's TEST-set metrics.

    Each result is a dict: {'epoch', 'MRR', 'Hit1', 'Hit3', 'Hit10'}.
    """
    by_epoch: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_epoch[row["epoch"]][row["split"]] = row

    def test_summary(epoch: int) -> dict:
        test_row = by_epoch[epoch]["test"]
        return {
            "epoch": epoch,
            "MRR": test_row["MRR"],
            "Hit1": test_row["Hit1"],
            "Hit3": test_row["Hit3"],
            "Hit10": test_row["Hit10"],
        }

    epochs_with_valid = [e for e, splits in by_epoch.items() if "valid" in splits and "test" in splits]
    epochs_with_test = [e for e, splits in by_epoch.items() if "test" in splits]

    # A log can have zero completed eval blocks (e.g. the run crashed or was
    # killed -- OOM, walltime, disconnect -- before its first --valid_epoch).
    # That is a legitimate log shape, not an error: report "no eval yet"
    # instead of letting max() raise on an empty sequence.
    if not epochs_with_valid or not epochs_with_test:
        return {"best_by_valid": None, "best_by_test": None}

    best_valid_epoch = max(epochs_with_valid, key=lambda e: by_epoch[e]["valid"]["MRR"])
    best_test_epoch = max(epochs_with_test, key=lambda e: by_epoch[e]["test"]["MRR"])

    return {
        "best_by_valid": test_summary(best_valid_epoch),
        "best_by_test": test_summary(best_test_epoch),
    }


def epoch_times(path: str) -> list[tuple]:
    """Parse per-epoch training lines into (epoch, loss, cumulative_seconds)."""
    result = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = EPOCH_LINE_RE.search(line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                seconds = float(match.group(3))
                result.append((epoch, loss, seconds))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize MyGO training logs.")
    parser.add_argument("logs", nargs="+", help="Path(s) to MyGO log file(s).")
    args = parser.parse_args(argv)

    print("| Log | Best-by | Epoch | MRR (%) | Hit@1 (%) | Hit@3 (%) | Hit@10 (%) |")
    print("|---|---|---|---|---|---|---|")
    for log_path in args.logs:
        rows = parse_log(log_path)
        s = summarize(rows)
        for label in ("best_by_valid", "best_by_test"):
            m = s[label]
            if m is None:
                print(f"| {log_path} | {label} | - | chưa có lượt đánh giá (no eval yet) | | | |")
                continue
            print(
                f"| {log_path} | {label} | {m['epoch']} "
                f"| {m['MRR'] * 100:.2f} | {m['Hit1'] * 100:.2f} "
                f"| {m['Hit3'] * 100:.2f} | {m['Hit10'] * 100:.2f} |"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
