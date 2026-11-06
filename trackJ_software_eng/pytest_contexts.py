"""Pytest plugin recording each collected item as a coverage context."""
import json
import os


def _set_context(item):
    try:
        from coverage import Coverage
        cov = Coverage.current()
        if cov is not None:
            cov.switch_context(item.nodeid)
    except (ImportError, AttributeError):
        pass


def pytest_runtest_setup(item):
    _set_context(item)


def pytest_runtest_logreport(report):
    """Emit a per-phase fallback if pytest's duration table omits a case."""
    if os.environ.get("TRACKJ_REPORT_DURATIONS") == "1":
        print("TRACKJ_DURATION " + json.dumps({"nodeid": report.nodeid, "phase": report.when,
                                               "seconds": report.duration}), flush=True)
