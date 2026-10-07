"""County hot spots of respiratory ED share: Getis-Ord Gi* and local Moran's I.

The question a hot-spot test answers is not "which counties are high" (a
choropleth shows that) but "where are high counties *surrounded by* high
counties more than chance would arrange". That is what separates a regional
outbreak from one noisy county with a small ED.

Choices, and why
----------------
* **Queen contiguity on the lower 48.** Counties sharing an edge or a corner
  are neighbors. Alaska and Hawaii have no land neighbors in the study area,
  so they are excluded rather than given made-up ones.
* **Islands are attached to their nearest neighbor.** After dropping counties
  with no data, a few counties are left with no reporting neighbors (and real
  islands like Nantucket have none). A zero-neighbor county gets a
  meaningless Gi*, so each is linked to its single nearest centroid.
* **Row-standardized weights**, so a county with 12 neighbors does not get a
  bigger statistic than one with 4 merely for having more.
* **Permutation p-values, then Benjamini-Hochberg FDR.** Testing ~2,400
  counties at alpha = 0.05 would flag ~120 by chance alone; the FDR
  correction is what makes a "hot spot" label mean something. The default
  9,999 permutations matter: with 999 the smallest possible pseudo p-value
  is 0.001, and after FDR across ~2,400 tests nothing can reach 99%.
  P-values are two-sided (high *or* low is the alternative), the choice
  esda recommends and will make its default; the sign of z then says which.
* **Respiratory share = COVID % + influenza % + RSV %.** All three are
  shares of the same denominator (all ED visits in the county-week), so they
  add. A visit coded for two of them is counted twice; that overlap is small
  and NSSP does not publish it, so we accept the slight overstatement.
"""

from __future__ import annotations

import warnings

import esda
import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import KNN, Queen
from libpysal.weights.util import attach_islands, fill_diagonal
from scipy.stats import false_discovery_control

from src.snapshot import SnapshotStore, coerce_date
from src.spatial.shapes import EQUAL_AREA_CRS, lower48

__all__ = [
    "SIGNALS", "respiratory_share", "week_values", "contiguity_weights",
    "gi_star", "local_moran", "LEVELS",
]

SIGNALS = {"covid": "nssp_covid", "influenza": "nssp_influenza", "rsv": "nssp_rsv"}

# Significance levels reported on the map, strictest first.
LEVELS = (0.01, 0.05, 0.10)


def respiratory_share(store: SnapshotStore, as_of=None) -> pd.DataFrame:
    """Per county-week COVID, flu, RSV and combined respiratory % of ED visits.

    ``as_of=None`` reads the latest vintage: fine for a map, never for
    scoring. Pass a date to get the series exactly as it was known that day.
    A county-week is kept only if all three signals are present, so the sum
    never quietly mixes a revised COVID value with a missing RSV one.
    """
    frames = []
    for name, source in SIGNALS.items():
        vintage = (store.latest_vintage(source) if as_of is None
                   else store.load_vintage(source, as_of))
        frames.append(vintage[["geo_value", "time_value", "value"]]
                      .rename(columns={"value": name})
                      .set_index(["geo_value", "time_value"]))
    wide = pd.concat(frames, axis=1, join="inner").reset_index()
    wide = wide.rename(columns={"geo_value": "fips", "time_value": "week_start"})
    wide["week_end"] = wide["week_start"] + pd.Timedelta(days=6)
    wide["resp_pct"] = wide[list(SIGNALS)].sum(axis=1, min_count=len(SIGNALS))
    return wide.dropna(subset=["resp_pct"]).reset_index(drop=True)


def week_values(share: pd.DataFrame, week_end) -> pd.DataFrame:
    """One week's rows of :func:`respiratory_share`, identified by its Saturday."""
    week_end = coerce_date(week_end)
    if week_end.weekday() != 5:
        raise ValueError(f"week_end must be a Saturday (MMWR week end), got "
                         f"{week_end.date()} ({week_end.day_name()})")
    return share[share["week_end"] == week_end].reset_index(drop=True)


def contiguity_weights(shapes: gpd.GeoDataFrame):
    """Queen weights keyed by ``fips``, with islands tied to their nearest county.

    Returns ``(w, islands)`` where ``islands`` lists the FIPS that had no
    contiguous neighbor before attachment, so callers can report them.
    """
    frame = shapes.set_index("fips")
    with warnings.catch_warnings():
        # libpysal warns about islands and disconnected components; we handle
        # both explicitly below, so the warning is noise.
        warnings.simplefilter("ignore", UserWarning)
        w = Queen.from_dataframe(frame, use_index=True)
        islands = list(w.islands)
        if islands:
            points = gpd.GeoDataFrame(
                geometry=frame.geometry.to_crs(EQUAL_AREA_CRS).centroid,
                index=frame.index)
            w = attach_islands(w, KNN.from_dataframe(points, k=1, use_index=True))
    return w, islands


def _prepare(shapes: gpd.GeoDataFrame, values: pd.DataFrame, column: str,
             conus_only: bool):
    data = shapes.merge(values[["fips", column]], on="fips", how="inner")
    data = data.dropna(subset=[column])
    if conus_only:
        data = lower48(data)
    if len(data) < 3:
        raise ValueError(f"need at least 3 counties with {column!r}, got {len(data)}")
    data = data.sort_values("fips").reset_index(drop=True)
    w, islands = contiguity_weights(data)
    # Align y to the weights' id order; libpysal does not reorder for us.
    y = data.set_index("fips").loc[w.id_order, column].to_numpy(dtype=float)
    return w, y, islands


def _fdr(p: np.ndarray) -> np.ndarray:
    return false_discovery_control(np.clip(p, 0, 1), method="bh")


def _gi_class(z: float, p: float) -> str:
    for level in LEVELS:
        if p < level:
            kind = "Hot" if z > 0 else "Cold"
            return f"{kind} spot {int(round((1 - level) * 100))}%"
    return "Not significant"


def gi_star(shapes: gpd.GeoDataFrame, values: pd.DataFrame,
            column: str = "resp_pct", permutations: int = 9999,
            seed: int = 4109, conus_only: bool = True) -> pd.DataFrame:
    """Getis-Ord Gi* hot/cold spots with FDR-corrected permutation p-values.

    ``values`` needs ``fips`` and ``column``. Returns one row per analyzed
    county: ``fips, value, z, p, p_fdr, class, island``. ``class`` is e.g.
    "Hot spot 99%" when the FDR-adjusted p is below 0.01 and z > 0.
    """
    w, y, islands = _prepare(shapes, values, column, conus_only)
    # Gi* counts the county itself as one of its own neighbors. Setting the
    # diagonal to 1 *before* row-standardizing gives it the same weight as
    # each contiguous neighbor, the textbook definition, instead of letting
    # esda guess a self-weight.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        w_self = fill_diagonal(w, 1.0)
        g = esda.G_Local(y, w_self, transform="R", star=None,
                         permutations=permutations, seed=seed, n_jobs=1,
                         alternative="two-sided")
    # z_sim is the z-score against the permutation distribution, so z and p
    # describe the same test.
    out = pd.DataFrame({"fips": w.id_order, "value": y,
                        "z": g.z_sim, "p": g.p_sim})
    out["p_fdr"] = _fdr(out["p"].to_numpy())
    out["class"] = [_gi_class(z, p) for z, p in zip(out["z"], out["p_fdr"])]
    out["island"] = out["fips"].isin(islands)
    return out.sort_values("fips").reset_index(drop=True)


_QUADRANTS = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}


def local_moran(shapes: gpd.GeoDataFrame, values: pd.DataFrame,
                column: str = "resp_pct", permutations: int = 9999,
                seed: int = 4109, alpha: float = 0.05,
                conus_only: bool = True) -> pd.DataFrame:
    """Local Moran's I (LISA) clusters and outliers, FDR-corrected.

    Unlike Gi*, LISA separates clusters (High-High, Low-Low) from spatial
    outliers (High-Low: a hot county among cool neighbors), which is the
    pattern a single-county outbreak would show.
    """
    w, y, islands = _prepare(shapes, values, column, conus_only)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        lisa = esda.Moran_Local(y, w, transformation="r",
                                permutations=permutations, seed=seed, n_jobs=1,
                                alternative="two-sided")
    out = pd.DataFrame({"fips": w.id_order, "value": y,
                        "z": lisa.z_sim, "p": lisa.p_sim, "I": lisa.Is,
                        "quadrant": lisa.q})
    out["p_fdr"] = _fdr(out["p"].to_numpy())
    out["class"] = np.where(out["p_fdr"] < alpha,
                            out["quadrant"].map(_QUADRANTS), "Not significant")
    out["island"] = out["fips"].isin(islands)
    return out.sort_values("fips").reset_index(drop=True)
