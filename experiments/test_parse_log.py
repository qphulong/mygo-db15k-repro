from parse_log import parse_log, summarize, epoch_times

LOG = "fixtures/sample.log"
NO_EVAL_LOG = "fixtures/no_eval.log"


def test_rows():
    rows = parse_log(LOG)
    assert [(r["epoch"], r["split"]) for r in rows] == [
        (50, "valid"),
        (50, "test"),
        (100, "valid"),
        (100, "test"),
    ]


def test_row_fields_are_correct_types_and_values():
    rows = parse_log(LOG)
    valid50 = rows[0]
    assert valid50["epoch"] == 50
    assert valid50["split"] == "valid"
    assert abs(valid50["MR"] - 145.70000000001) < 1e-6
    assert abs(valid50["MRR"] - 0.301234) < 1e-9
    assert abs(valid50["Hit10"] - 0.5523) < 1e-9
    assert abs(valid50["Hit3"] - 0.3821) < 1e-9
    assert abs(valid50["Hit1"] - 0.2015) < 1e-9


def test_best_by_valid_uses_test_metrics_at_best_valid_epoch():
    s = summarize(parse_log(LOG))
    assert s["best_by_valid"]["epoch"] == 50
    assert abs(s["best_by_valid"]["MRR"] - 0.31) < 1e-9


def test_best_by_test():
    s = summarize(parse_log(LOG))
    assert s["best_by_test"]["epoch"] == 100
    assert abs(s["best_by_test"]["MRR"] - 0.33) < 1e-9


def test_epoch_times():
    t = epoch_times(LOG)
    assert t[0][0] == 1 and t[-1][0] == 100


def test_epoch_times_values():
    t = epoch_times(LOG)
    epochs = [e for e, _, _ in t]
    assert epochs == [1, 2, 50, 51, 100]
    first = t[0]
    assert first == (1, 812.345678, 9.876543)


def test_summarize_returns_none_placeholders_when_no_eval_blocks():
    # A log crashed/killed before its first --valid_epoch (e.g. OOM on the
    # very first training step) has zero completed eval blocks. summarize()
    # must not raise (previously: ValueError: max() iterable argument is
    # empty) -- it should report "no eval yet" via None placeholders.
    rows = parse_log(NO_EVAL_LOG)
    assert rows == []
    s = summarize(rows)
    assert s == {"best_by_valid": None, "best_by_test": None}
