"""The revision correction may only read vintages published on or before its origin."""

import numpy as np
import pandas as pd

from src import improve


def _table(revise_after: float) -> pd.DataFrame:
    """Two study series; each week first printed at lag 1, revised later by a known factor."""
    rows = []
    origins = pd.date_range("2025-01-04", periods=40, freq="7D")
    for fips in ("17031", "18097"):
        for o in origins:
            for lag in range(0, 53):
                week = o - pd.Timedelta(weeks=lag)
                if week < origins[0] - pd.Timedelta(weeks=52) or lag == 0:
                    continue
                true = 2.0 + np.sin(week.dayofyear / 58.0)
                factor = 0.9 if lag <= 2 else 1.0              # first prints 10% low
                if o > origins[25]:
                    factor = revise_after if lag <= 2 else 1.0
                rows.append({"fips": fips, "metro": "chi", "as_of": o, "week_end": week, "lag": lag,
                             "ed_resp": true * factor, "ed_resp_final": true, "ed_resp_first": true * factor})
    return pd.DataFrame(rows)


def test_correction_recovers_known_revision_and_never_reads_the_future():
    origin = pd.date_range("2025-01-04", periods=40, freq="7D")[20]
    a = improve.corrections(improve.revision_table(_table(0.9)), origin)
    b = improve.corrections(improve.revision_table(_table(0.5)), origin)   # later vintages differ
    assert abs(a[1] - np.log(1 / 0.9)) < 1e-9 and a.get(3, 0.0) == 0.0
    assert a == b


def test_slope_is_per_week_change():
    logs = np.log(np.array([[1.0, 2.0, 4.0, 8.0]]))
    assert np.isclose(improve._slope(logs)[0], np.log(2.0))
