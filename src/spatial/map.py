"""Interactive county map of respiratory ED share, with Gi* hot spots.

    python -m src.spatial.map --week 2025-12-27 --out demo/map.html

writes two self-contained HTML files: the national map (``demo/map.html``)
and a zoomed St. Louis region view (``demo/map_stl.html``). Basemap tiles
load from CARTO at view time; everything else is inlined.

What the colors mean
--------------------
By default each county is colored by its **first print**: the earliest value
published for that week (normally one week after it ended), i.e. what a
nowcaster actually saw. The tooltip
puts that next to the latest value so the revision is visible. Gi* is run on
the same values that are colored, so outlines and colors always agree.

First prints come from a separate store (``raw/snapshots_national``) holding
one all-county historical vintage per mapped week. It is separate from
``raw/snapshots`` on purpose: the study pipeline writes study-county-only
historical vintages there, and an all-county file for the same issue date
would collide with them. ``--fetch`` pulls the missing issues from Delphi
(3 small requests); without them the map falls back to the latest values
and says so in the subtitle.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from src.ingest_nssp import API, _UA, mmwr_week_start
from src.snapshot import SnapshotStore, coerce_date
from src.spatial.hotspots import SIGNALS, gi_star, respiratory_share, week_values
from src.spatial.shapes import load_counties, lower48
from src.study import METROS, week_end_of

__all__ = ["epiweek_of", "fetch_early_issues", "first_print_share", "map_frame", "build_map",
           "RAMP", "HOT", "STL_VIEW"]

# One-hue sequential ramp (light = low), from the project's dataviz palette.
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
HOT = "#e8590c"           # hot-spot outline: warm, far from every ramp step
COLD = "#5f6b7a"          # cold-spot outline (dashed), recessive
STUDY = "#111111"
ESRI_GRAY = ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
             "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}")

# Map view and the counties drawn in the zoomed regional file.
STL_VIEW = {"center": (38.63, -90.35), "zoom": 8,
            "bbox": (-92.6, 37.3, -88.6, 39.9)}   # lon/lat: MO-IL around STL


def epiweek_of(day) -> int:
    """MMWR epiweek (e.g. 202552) containing ``day``. Week 1 holds January 4."""
    start = coerce_date(day).date()
    start -= timedelta(days=(start.weekday() + 1) % 7)       # back to Sunday
    year = (start + timedelta(days=3)).year                  # Wednesday's year
    return year * 100 + (start - mmwr_week_start(year * 100 + 1)).days // 7 + 1


def fetch_early_issues(store: SnapshotStore, week_end, max_lag: int = 4,
                       api_key: str | None = None) -> int:
    """Store every issue of one week published within ``max_lag`` weeks.

    One request per signal using Delphi's ``issues`` range, rather than one
    ``as_of`` request per lag: anonymous access is rate-limited (HTTP 429
    with a retry-after of most of an hour), so 3 calls beats 12. Each Delphi
    issue is written as its own snapshot, stamped like ``ingest`` stamps
    as-of vintages (the Saturday ending the issue epiweek), so
    ``load_vintage`` semantics are unchanged. Returns rows written.
    """
    week_end = week_end_of(week_end)
    week = epiweek_of(week_end)
    issues = (f"{epiweek_of(week_end + pd.Timedelta(days=7))}-"
              f"{epiweek_of(week_end + pd.Timedelta(days=7 * max_lag))}")
    written = 0
    for name, source in SIGNALS.items():
        params = {"data_source": "nssp", "signal": f"pct_ed_visits_{name}",
                  "time_type": "week", "geo_type": "county", "geo_values": "*",
                  "time_values": week, "issues": issues}
        if api_key:
            params["api_key"] = api_key
        response = requests.get(API, params=params, timeout=120, headers=_UA)
        response.raise_for_status()
        payload = response.json()
        if payload.get("result") == -2:
            continue
        if payload.get("result") != 1:
            raise RuntimeError(f"{source} {week}: {payload.get('message')}")
        raw = pd.DataFrame(payload["epidata"])
        for delphi_issue, rows in raw.groupby("issue"):
            stamp = pd.Timestamp(mmwr_week_start(int(delphi_issue)) + timedelta(days=6))
            store.write_snapshot(source, stamp, pd.DataFrame({
                "geo_value": rows["geo_value"].astype(str).str.zfill(5),
                "time_value": pd.Timestamp(mmwr_week_start(week)),
                "value": pd.to_numeric(rows["value"], errors="coerce"),
                "issue": stamp,
                "delphi_issue": rows["issue"].astype("int64"),
            }))
            written += len(rows)
    return written


def first_print_share(week_end, root="raw/snapshots_national", fetch: bool = False,
                      max_lag: int = 4, api_key: str | None = None
                      ) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Respiratory share for one week as each county was first published.

    "First print" is per county and per signal: the earliest stored issue
    within ``max_lag`` weeks. It is usually one week after the week ends, but
    Delphi's archive has holes: week 202552 (ending 2025-12-27) has nothing
    at as-of 202553 and first appears at 202601. Returns the frame and the
    latest issue date any of those first prints carries, i.e. the date by
    which all of it was public.
    """
    week_end = week_end_of(week_end)
    week_start = week_end - pd.Timedelta(days=6)
    horizon = week_end + pd.Timedelta(days=7 * max_lag)
    store = SnapshotStore(root)
    def window(source):
        return [i for i in store.available_issues(source) if week_end < i <= horizon]

    if fetch and not all(window(s) for s in SIGNALS.values()):
        fetch_early_issues(store, week_end, max_lag, api_key)

    firsts = []
    for name, source in SIGNALS.items():
        rows = [store.load_vintage(source, i) for i in window(source)]
        rows = [r[r["time_value"] == week_start] for r in rows if not r.empty]
        if not rows:
            return pd.DataFrame(columns=["fips", "week_end", "resp_pct"]), None
        early = (pd.concat(rows).sort_values("issue")
                 .drop_duplicates("geo_value", keep="first"))
        firsts.append(early.set_index("geo_value")[["value", "issue"]]
                      .rename(columns={"value": name, "issue": f"{name}_issue"}))
    wide = pd.concat(firsts, axis=1, join="inner").reset_index(names="fips")
    wide["week_end"] = week_end
    wide["resp_pct"] = wide[list(SIGNALS)].sum(axis=1, min_count=len(SIGNALS))
    known = wide[[f"{n}_issue" for n in SIGNALS]].max().max()
    return wide.dropna(subset=["resp_pct"]).reset_index(drop=True), known


def map_frame(counties: gpd.GeoDataFrame, latest: pd.DataFrame,
              first: pd.DataFrame, use_first: bool = True,
              permutations: int = 9999) -> gpd.GeoDataFrame:
    """Join shapes, first/latest values and Gi* classes into one frame.

    ``value`` is what gets colored and tested: the first print where one
    exists (``use_first``), otherwise the latest value.
    """
    out = counties.merge(latest[["fips", "resp_pct"]].rename(columns={"resp_pct": "latest"}),
                         on="fips", how="left")
    out = out.merge(first[["fips", "resp_pct"]].rename(columns={"resp_pct": "first"}),
                    on="fips", how="left")
    out["value"] = out["first"] if use_first else out["latest"]
    scored = out.dropna(subset=["value"])
    gi = gi_star(counties, scored[["fips", "value"]], column="value",
                 permutations=permutations)
    out = out.merge(gi[["fips", "z", "p_fdr", "class"]], on="fips", how="left")
    out["class"] = out["class"].fillna("No data")
    return out


def _bins(values: pd.Series, k: int = len(RAMP)) -> list[float]:
    """Quantile class breaks rounded to 0.5 points, deduplicated.

    Quantiles keep every color in use whatever the week's level (2% off
    season, 10%+ at peak); fixed breaks would paint a summer week one color.
    """
    qs = np.nanquantile(values, np.linspace(0, 1, k + 1))
    edges = sorted({round(q * 2) / 2 for q in qs[1:-1]})
    return [0.0] + [e for e in edges if e > 0] + [float(np.ceil(values.max()))]


def _fmt(x) -> str:
    return "–" if pd.isna(x) else f"{x:.1f}%"


def _round_coords(obj, nd: int = 3):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, (list, tuple)):
        return [_round_coords(o, nd) for o in obj]
    if isinstance(obj, dict):
        return {k: _round_coords(v, nd) for k, v in obj.items()}
    return obj


def _geojson(frame: gpd.GeoDataFrame, colormap, decimals: int) -> dict:
    """Compact GeoJSON with display-ready properties (strings, fill color)."""
    props = pd.DataFrame({
        "fips": frame["fips"],
        "county": frame["name"] + ", " + frame["state"],
        "value": frame["value"].map(_fmt),
        "first": frame["first"].map(_fmt),
        "latest": frame["latest"].map(_fmt),
        "revision": [("–" if pd.isna(f) or pd.isna(l) or f == 0
                      else f"{(l - f) / f:+.0%}")
                     for f, l in zip(frame["first"], frame["latest"])],
        "gi": frame["class"],
        "fill": [("#e6e4df" if pd.isna(v) else colormap(v)) for v in frame["value"]],
    })
    gj = gpd.GeoDataFrame(props, geometry=frame.geometry.values,
                          crs=frame.crs).__geo_interface__
    for feature in gj["features"]:
        feature.pop("bbox", None)
        feature["geometry"]["coordinates"] = _round_coords(
            feature["geometry"]["coordinates"], decimals)
    gj.pop("bbox", None)
    return gj


def _outline(frame: gpd.GeoDataFrame, mask, name: str, color: str,
             weight: float, dash: str | None = None, show: bool = True):
    selected = frame.loc[mask]
    if selected.empty:
        return None
    merged = gpd.GeoDataFrame(geometry=[selected.geometry.union_all()], crs=frame.crs)
    style = {"color": color, "weight": weight, "fill": False, "opacity": 0.95}
    if dash:
        style["dashArray"] = dash
    return folium.GeoJson(merged.__geo_interface__, name=name,
                          style_function=lambda _f, s=style: s,
                          interactive=False, show=show)


def _legend_html(bins: list[float]) -> str:
    """Equal-width swatches: a step legend, not a proportional bar.

    branca's colorbar spaces ticks by value, so one 50% outlier squeezes all
    other classes into a sliver; class legends read better as even steps.
    """
    cells = []
    for i, color in enumerate(RAMP[:len(bins) - 1]):
        lo, hi = bins[i], bins[i + 1]
        label = (f"&lt;{hi:g}" if i == 0 else
                 f"{lo:g}+" if i == len(bins) - 2 else f"{lo:g}–{hi:g}")
        cells.append(f'<div style="flex:1;text-align:center"><div style="height:12px;'
                     f'background:{color};margin:0 1px"></div>{label}</div>')
    cells.append('<div style="flex:1;text-align:center"><div style="height:12px;'
                 'background:#e6e4df;margin:0 1px 0 8px"></div>no data</div>')
    return ('<div style="margin-top:8px;color:#333">Respiratory % of ED visits '
            '(COVID + flu + RSV)</div><div style="display:flex;font-size:11px;'
            f'color:#555;margin-top:3px">{"".join(cells)}</div>')


def _overlay_html(title: str, subtitle: str, credit: str, n_hot: int,
                  bins: list[float]) -> str:
    return f"""
<div style="position:fixed;top:12px;left:56px;z-index:9999;background:rgba(255,255,255,.95);
  padding:10px 14px 12px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.25);
  font:13px/1.35 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#222;width:470px">
  <div style="font-size:17px;font-weight:600">{title}</div>
  <div style="color:#555;margin-top:2px">{subtitle}</div>
  {_legend_html(bins)}
  <div style="margin-top:8px;display:flex;gap:16px;align-items:center;color:#333">
    <span><svg width="22" height="12"><rect x="1" y="1" width="20" height="10" fill="none"
      stroke="{HOT}" stroke-width="3"/></svg> Gi* hot spot ({n_hot} {"county" if n_hot == 1 else "counties"} shown, FDR&lt;0.10)</span>
    <span><svg width="22" height="12"><rect x="1" y="1" width="20" height="10" fill="none"
      stroke="{STUDY}" stroke-width="2.5"/></svg> Study county</span>
  </div>
</div>
<div style="position:fixed;bottom:4px;left:50%;transform:translateX(-50%);z-index:9999;
  background:rgba(255,255,255,.85);padding:3px 8px;border-radius:4px;
  font:11px -apple-system,Helvetica,Arial,sans-serif;color:#555;white-space:nowrap">
  {credit}</div>"""


def build_map(frame: gpd.GeoDataFrame, *, title: str, subtitle: str,
              center=(38.5, -96.5), zoom: int = 5, decimals: int = 3,
              label_metros: bool = True, basemap: bool = False,
              bins: list[float] | None = None, label_counties: bool = False,
              permutations: int = 9999) -> folium.Map:
    """A folium map: choropleth, hot/cold outlines, study counties, legend.

    Pass the national ``bins`` to a regional map so a color means the same
    value in both views.
    """
    bins = bins or _bins(frame["value"].dropna())
    colormap = cm.StepColormap(RAMP[:len(bins) - 1], index=bins,
                               vmin=bins[0], vmax=bins[-1])

    fmap = folium.Map(location=center, zoom_start=zoom, tiles=None,
                      control_scale=True, prefer_canvas=True)
    # No basemap nationally: the counties tile the whole view, so tiles add
    # only bytes and a network dependency (CARTO now wants an API key). The
    # regional view gets Esri's keyless light-gray canvas for river/road context.
    if basemap:
        folium.TileLayer(ESRI_GRAY, attr="Tiles &copy; Esri", name="Basemap",
                         control=False).add_to(fmap)
    fmap.get_root().header.add_child(folium.Element(
        "<style>.leaflet-container{background:#f7f6f3 !important}</style>"))

    folium.GeoJson(
        _geojson(frame, colormap, decimals), name="Respiratory %",
        style_function=lambda f: {"fillColor": f["properties"]["fill"],
                                  "fillOpacity": 0.88, "color": "#ffffff",
                                  "weight": 0.3},
        highlight_function=lambda _f: {"weight": 2, "color": "#333"},
        tooltip=folium.GeoJsonTooltip(
            fields=["county", "fips", "value", "first", "latest", "revision", "gi"],
            aliases=["County", "FIPS", "Mapped value", "First print",
                     "Latest", "Revision since", "Gi*"],
            sticky=True),
    ).add_to(fmap)

    states = frame[["state", "geometry"]].dissolve("state")
    folium.GeoJson(gpd.GeoDataFrame(geometry=states.boundary.values, crs=frame.crs).__geo_interface__,
                   name="State borders", interactive=False, control=False,
                   style_function=lambda _f: {"color": "#ffffff", "weight": 1.4,
                                              "opacity": 0.9}).add_to(fmap)

    hot = frame["class"].str.startswith("Hot")
    cold = frame["class"].str.startswith("Cold")
    for layer in (_outline(frame, hot, "Gi* hot spots", HOT, 2.5),
                  _outline(frame, cold, "Gi* cold spots", COLD, 1.5, "4 4", show=False)):
        if layer is not None:
            layer.add_to(fmap)

    study_names = {f: m.name for m in METROS.values() for f in m.counties}
    study = frame["fips"].isin(study_names)
    layer = _outline(frame, study, "Study counties", STUDY, 2.5)
    if layer is not None:
        layer.add_to(fmap)
    if label_metros:
        labels = folium.FeatureGroup(name="Metro labels")
        for metro in METROS.values():
            rows = frame[frame["fips"].isin(metro.counties)]
            if rows.empty:
                continue
            lat, lon = rows["lat"].mean(), rows["lon"].max() + 0.25
            folium.Marker(
                (lat, lon),
                icon=folium.DivIcon(
                    icon_size=(140, 18), icon_anchor=(0, 9),
                    html=f'<div style="font:600 12px -apple-system,Helvetica,Arial;'
                         f'color:#111;text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff;'
                         f'white-space:nowrap">{metro.name}</div>'),
            ).add_to(labels)
        labels.add_to(fmap)

    if label_counties:
        names = folium.FeatureGroup(name="County names")
        for row in frame.itertuples():
            if row.fips in study_names:
                continue        # the metro label already names them
            short = row.name.removesuffix(" County")
            # Ink by background: white on the four darkest steps, dark otherwise.
            dark = not pd.isna(row.value) and RAMP.index(colormap(row.value)[:7]) >= 3
            ink = "color:#fff" if dark else "color:#333;text-shadow:0 0 2px #fff"
            folium.Marker(
                (row.lat, row.lon),
                icon=folium.DivIcon(
                    icon_size=(110, 14), icon_anchor=(55, 7),
                    html=f'<div style="font:10px -apple-system,Helvetica,Arial;'
                         f'text-align:center;{ink}">{short}</div>'),
            ).add_to(names)
        names.add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    credit = ("Data: CDC NSSP via Delphi Epidata · Boundaries: US Census 2021 "
              "cartographic files · Hot spots: Getis-Ord Gi*, Queen contiguity, "
              f"{permutations:,} permutations, Benjamini-Hochberg FDR")
    fmap.get_root().html.add_child(folium.Element(
        _overlay_html(title, subtitle, credit, int(hot.sum()), bins)))
    return fmap


def _region(frame: gpd.GeoDataFrame, bbox) -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = bbox
    return frame.cx[xmin:xmax, ymin:ymax]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the county hot-spot map.")
    parser.add_argument("--week", required=True,
                        help="week-ending Saturday, e.g. 2025-12-27 (any day in the week works)")
    parser.add_argument("--store", default="raw/snapshots")
    parser.add_argument("--first-print-store", default="raw/snapshots_national")
    parser.add_argument("--fetch", action="store_true",
                        help="pull the first-print vintage from Delphi if missing")
    parser.add_argument("--api-key", default=None, help="free Delphi key (avoids 429s)")
    parser.add_argument("--color-by", choices=("first", "latest"), default="first")
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--out", default="demo/map.html")
    args = parser.parse_args()

    week_end = week_end_of(args.week)
    store = SnapshotStore(args.store)
    latest_issue = store.available_issues("nssp_influenza")[-1]
    latest = week_values(respiratory_share(store), week_end)
    if latest.empty:
        raise SystemExit(f"no NSSP data for the week ending {week_end.date()}")
    first, first_issue = first_print_share(week_end, args.first_print_store,
                                           args.fetch, api_key=args.api_key)
    use_first = args.color_by == "first" and not first.empty

    week_label = f"{week_end:%b} {week_end.day}, {week_end.year}"
    if use_first:
        known = first_issue
        subtitle = (f"As first published ({known:%b} {known.day}, {known.year}) — what a "
                    f"nowcaster saw. Latest values ({latest_issue.date()}) in tooltips.")
    else:
        subtitle = (f"Latest values as of {latest_issue.date()} (first print unavailable; "
                    f"run with --fetch).")
    title = f"Respiratory ED visits, week ending {week_label}"

    # National: coarse 20m shapes keep the file small; Gi* still uses the
    # detailed 500k shapes for contiguity, so the statistic does not depend
    # on display simplification.
    detailed = lower48(load_counties("500k"))
    frame = map_frame(detailed, latest, first, use_first, args.permutations)
    coarse = lower48(load_counties("20m"))
    national = coarse[["fips", "geometry"]].merge(
        frame.drop(columns="geometry"), on="fips", how="inner")
    national = gpd.GeoDataFrame(national, geometry="geometry", crs=coarse.crs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bins = _bins(frame["value"].dropna())
    build_map(national, title=title, subtitle=subtitle, bins=bins,
              permutations=args.permutations).save(out)

    regional = _region(frame, STL_VIEW["bbox"])
    regional = regional.assign(geometry=regional.geometry.simplify(0.002))
    stl_out = out.with_name(out.stem + "_stl" + out.suffix)
    build_map(regional, title=f"St. Louis region · week ending {week_label}",
              subtitle=subtitle, center=STL_VIEW["center"], zoom=STL_VIEW["zoom"],
              decimals=4, basemap=True, bins=bins, label_counties=True,
              permutations=args.permutations).save(stl_out)

    hot = frame["class"].str.startswith("Hot")
    print(f"week ending {week_end.date()} · colored by "
          f"{'first print' if use_first else 'latest'} · {int(hot.sum())} hot-spot counties")
    for fips, name in [(f, n) for m in METROS.values() for f, n in m.counties.items()]:
        row = frame.loc[frame["fips"] == fips].iloc[0]
        print(f"  {fips} {name:<18} {_fmt(row['value']):>6}  {row['class']}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), "
          f"{stl_out} ({stl_out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
