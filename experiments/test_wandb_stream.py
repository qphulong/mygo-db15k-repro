import threading
import time

from wandb_stream import LogStreamParser, follow_log

LOG = "fixtures/sample.log"


def parse_all(path):
    p = LogStreamParser()
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            events.extend(p.feed(line))
    return events


def test_epoch_lines_emit_loss_and_epoch_time():
    events = parse_all(LOG)
    train = [e for e in events if "train/loss" in e]
    assert [e["epoch"] for e in train] == [1, 2, 50, 51, 100]
    assert train[0] == {"epoch": 1, "train/loss": 812.345678,
                        "train/elapsed_s": 9.876543, "train/epoch_time_s": 9.876543}
    # second epoch: time of this epoch only, not cumulative
    assert abs(train[1]["train/epoch_time_s"] - 9.876543) < 1e-6


def test_eval_blocks_emit_one_event_per_split_in_percent():
    events = parse_all(LOG)
    evals = [e for e in events if "train/loss" not in e]
    assert [(e["epoch"], sorted(k.split("/")[0] for k in e if k != "epoch")[0]) for e in evals] == [
        (50, "valid"), (50, "test"), (100, "valid"), (100, "test"),
    ]
    v50 = evals[0]
    assert abs(v50["valid/MRR"] - 30.1234) < 1e-9
    assert abs(v50["valid/Hit@10"] - 55.23) < 1e-9
    assert abs(v50["valid/Hit@3"] - 38.21) < 1e-9
    assert abs(v50["valid/Hit@1"] - 20.15) < 1e-9
    assert abs(v50["valid/MR"] - 145.70000000001) < 1e-9
    assert abs(evals[3]["test/MRR"] - 33.0) < 1e-9


def test_partial_eval_block_is_not_emitted():
    p = LogStreamParser()
    p.feed("2026-09-19 10:08:13,700 - INFO - 50 \t 120.4 \t 493.8 s\n")
    p.feed("2026-09-19 10:08:13,827 - INFO - Link Prediction on Validation Set\n")
    assert p.feed("2026-09-19 10:08:13,828 - INFO - MRR: 0.3\n") == []


class FakeProc:
    def __init__(self):
        self.done = False

    def poll(self):
        return 0 if self.done else None


def test_follow_log_reads_file_written_while_process_runs(tmp_path):
    log_dir = tmp_path / "logs"
    lines = open(LOG, encoding="utf-8").readlines()
    proc = FakeProc()

    def writer():
        time.sleep(0.05)  # log file appears only after the process starts
        log_dir.mkdir()
        with open(log_dir / "run.log", "w", encoding="utf-8") as f:
            for i, line in enumerate(lines):
                if i == 5:
                    f.write(line[:10])  # half-written line must be buffered
                    f.flush()
                    time.sleep(0.05)
                    f.write(line[10:])
                else:
                    f.write(line)
                f.flush()
        proc.done = True

    t = threading.Thread(target=writer)
    t.start()
    got = []
    path = follow_log(str(log_dir / "*.log"), proc, got.append, poll_s=0.01)
    t.join()
    assert path == str(log_dir / "run.log")
    assert got == lines


def test_follow_log_returns_none_when_process_dies_without_log(tmp_path):
    proc = FakeProc()
    proc.done = True
    assert follow_log(str(tmp_path / "*.log"), proc, lambda _: None, poll_s=0.01) is None
