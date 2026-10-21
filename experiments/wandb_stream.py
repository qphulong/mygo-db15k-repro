"""Stream a MyGO training log, while training is still running, into
W&B-ready metric dicts.

The author's training script only writes to its log file, so instead of
patching it we follow that file (like ``tail -f``) and turn each completed
line into an event:

- an epoch line  -> {"epoch", "train/loss", "train/elapsed_s", "train/epoch_time_s"}
- a complete eval block (5 metric lines after a "Link Prediction on ..." header)
  -> {"epoch", "<split>/MRR", "<split>/Hit@1", "<split>/Hit@3", "<split>/Hit@10",
      "<split>/MR"} with MRR/Hit@k in percent, split being "valid" or "test".

Standard library only, plus ``parse_log`` (its regexes are reused so both
parsers always agree). Inlined verbatim into the Kaggle notebook.
"""

from __future__ import annotations

import glob
import os
import time

from parse_log import EPOCH_LINE_RE, METRIC_LINE_RE, TEST_HEADER_RE, VALID_HEADER_RE

METRIC_KEYS = {"MR": "MR", "MRR": "MRR", "Hit1": "Hit@1", "Hit3": "Hit@3", "Hit10": "Hit@10"}


class LogStreamParser:
    """Feed log lines one at a time; ``feed`` returns the events they complete."""

    def __init__(self):
        self.last_epoch = None
        self.last_elapsed = 0.0
        self.split = None
        self.metrics = {}

    def feed(self, line: str) -> list[dict]:
        m = EPOCH_LINE_RE.search(line)
        if m:
            epoch, loss, elapsed = int(m.group(1)), float(m.group(2)), float(m.group(3))
            event = {"epoch": epoch, "train/loss": loss, "train/elapsed_s": elapsed,
                     "train/epoch_time_s": elapsed - self.last_elapsed}
            self.last_epoch, self.last_elapsed = epoch, elapsed
            return [event]

        if VALID_HEADER_RE.search(line):
            self.split, self.metrics = "valid", {}
            return []
        if TEST_HEADER_RE.search(line):
            self.split, self.metrics = "test", {}
            return []

        m = METRIC_LINE_RE.search(line)
        if m and self.split is not None:
            key, value = m.group(1), float(m.group(2))
            self.metrics[key] = value if key == "MR" else value * 100
            if len(self.metrics) == len(METRIC_KEYS):
                event = {"epoch": self.last_epoch}
                for key, value in self.metrics.items():
                    event[f"{self.split}/{METRIC_KEYS[key]}"] = value
                self.split, self.metrics = None, {}
                return [event]
        return []


def follow_log(pattern: str, proc, on_line, poll_s: float = 5.0):
    """Call ``on_line`` for every complete line of the log matching ``pattern``
    until ``proc`` (anything with a ``poll()`` method, e.g. ``subprocess.Popen``)
    has exited and the file is fully read.

    The log file may appear only after the process starts. Returns the log
    path, or None if the process exited without ever creating it.
    """
    path = None
    while path is None:
        matches = sorted(glob.glob(pattern))
        if matches:
            path = matches[0]
        elif proc.poll() is not None:
            return None
        else:
            time.sleep(poll_s)

    buf = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            exited = proc.poll() is not None  # check BEFORE reading, so nothing written after is lost
            chunk = f.read()
            if chunk:
                buf += chunk
                *lines, buf = buf.split("\n")
                for line in lines:
                    on_line(line + "\n")
            elif exited:
                break
            else:
                time.sleep(poll_s)
    if buf:
        on_line(buf)
    return os.path.normpath(path)
