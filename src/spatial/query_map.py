"""Draw a ``query.Answer``: a self-contained folium map with the written answer beside it.

Colors follow the project's map conventions (``src.spatial.map``): a one-hue
blue ramp with quantile classes for levels, so every class is used whatever
the season; for week-over-week change, a diverging scale -- blue for falling,
red for rising, the palette's neutral gray for "about the same" -- with the
red arm matched to the blue arm step for step in lightness, so neither
direction looks louder than the other. Both arms pass the dataviz validator
as ordinal ramps (monotone lightness, visible steps, light ends >= 2:1
against the surface).

Outlines: study counties in black, as on every project map. Gi* hot spots in
orange on level maps; clusters of rising (or falling) counties dashed in ink
on change maps, where an orange line would vanish against the red arm.

Every piece of text in the panel is escaped: questions are typed by the
audience, and article titles and quotes come from scraped web pages.
"""

from __future__ import annotations

import html

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd

from src.spatial.map import COLD, ESRI_GRAY, HOT, RAMP, STUDY, _bins, _outline, _round_coords
from src.spatial.query import SIGNALS, Answer, Gazetteer, summary_lines
from src.study import METROS

__all__ = ["DIVERGING", "change_bins", "render"]

DIVERGING = ("#184f95", "#3987e5", "#86b6ef", "#f0efec", "#ea9a93", "#d75853", "#892b2a")
NO_DATA = "#e6e4df"
INK = "#0b0b0b"


def _nice(x: float) -> float:
    """The round number (1, 2, 2.5 or 5 times a power of ten) nearest ``x``."""
    if not np.isfinite(x) or x <= 0:
        return 0.5
    exp = np.floor(np.log10(x))
    steps = np.array([1, 2, 2.5, 5, 10]) * 10 ** exp
    return float(max(steps[np.argmin(np.abs(np.log(steps / x)))], 0.1))


def change_bins(change: pd.Series) -> list[float]:
    """Symmetric breaks in points: |change| < b is "about the same", then a, 2a.

    ``a`` is a round number near half the 90th percentile of |change|, so the
    darkest steps are kept for the largest moves of that week; ``b`` is the
    round number nearest a/4.
    """
    moves = change.dropna().abs()
    a = _nice(float(np.quantile(moves, 0.9)) / 2) if len(moves) else 1.0
    b = min(_nice(a / 4), a / 2)
    return [-np.inf, -2 * a, -a, -b, b, a, 2 * a, np.inf]


def _fmt_pct(x) -> str:
    return "–" if pd.isna(x) else f"{x:.1f}%"


def _fmt_pts(x) -> str:
    if pd.isna(x):
        return "–"
    return "±0.0 pts" if round(x, 1) == 0 else f"{x:+.1f} pts"


def _colors(frame: pd.DataFrame, measure: str):
    """Fill color per county, and the legend cells [(color, label)]."""
    if measure == "level":
        values = frame["value"].dropna()
        bins = _bins(values) if len(values) else [0.0, 1.0]
        ramp = cm.StepColormap(RAMP[:len(bins) - 1], index=bins, vmin=bins[0], vmax=bins[-1])
        fills = [NO_DATA if pd.isna(v) else ramp(v) for v in frame["value"]]
        cells = []
        for i, color in enumerate(RAMP[:len(bins) - 1]):
            lo, hi = bins[i], bins[i + 1]
            cells.append((color, f"<{hi:g}" if i == 0 else f"{lo:g}+" if i == len(bins) - 2
                          else f"{lo:g}–{hi:g}"))
        return fills, cells, "% of ED visits", None
    bins = change_bins(frame["change"])
    idx = np.digitize(frame["change"].to_numpy(float), bins[1:-1])
    fills = [NO_DATA if pd.isna(v) else DIVERGING[i] for v, i in zip(frame["change"], idx)]
    ticks = [f"{b:+g}".replace("-", "−") for b in bins[1:-1]]   # boundaries between swatches
    return fills, [(c, "") for c in DIVERGING], "change vs the week before (percentage points)", ticks


def _geojson(frame: gpd.GeoDataFrame, fills, decimals: int) -> dict:
    props = pd.DataFrame({
        "county": frame["name"] + ", " + frame["state"],
        "value": frame["value"].map(_fmt_pct),
        "prev": frame["prev"].map(_fmt_pct),
        "change": frame["change"].map(_fmt_pts),
        "trend": (frame["trend"].fillna("–") if "trend" in frame else "–"),
        "cluster": frame["cluster"],
        "fill": fills,
    })
    gj = gpd.GeoDataFrame(props, geometry=frame.geometry.values, crs=frame.crs).__geo_interface__
    for feature in gj["features"]:
        feature.pop("bbox", None)
        feature["geometry"]["coordinates"] = _round_coords(feature["geometry"]["coordinates"], decimals)
    gj.pop("bbox", None)
    return gj


def _panel(ans: Answer, gaz: Gazetteer, cells, unit: str, ticks=None) -> str:
    q, text = ans.query, summary_lines(ans, gaz)
    e = html.escape
    chips = [SIGNALS[q.signal].split(" (")[0],
             "level" if q.measure == "level" else f"change {'↑' if q.direction == 'up' else '↓'}",
             f"week ending {q.week_end:%b} {q.week_end.day}, {q.week_end.year}",
             gaz.label(q.place), f"read by {q.parser}"]
    legend = "".join(f'<div class="cell"><div style="background:{c}"></div>{e(lab)}</div>'
                     for c, lab in cells)
    legend += f'<div class="cell"><div style="background:{NO_DATA}"></div>no data</div>'
    if ticks:            # diverging: label the boundaries, not the ranges
        width = 100 / (len(cells) + 1)
        legend += "".join(f'<span class="tick" style="left:{width * (i + 1):.2f}%">{e(t)}</span>'
                          for i, t in enumerate(ticks))
    outline = ("Gi* hot spot (FDR&lt;0.10)" if q.measure == "level"
               else f"cluster of {'rising' if q.direction == 'up' else 'falling'} counties (LISA, FDR&lt;0.05)")
    swatch = (f'<svg width="22" height="12"><rect x="1" y="1" width="20" height="10" fill="none" '
              f'stroke="{HOT if q.measure == "level" else INK}" stroke-width="2.5"'
              f'{"" if q.measure == "level" else " stroke-dasharray=" + chr(34) + "4 3" + chr(34)}/></svg>')
    places = "".join(f"<li>{e(p)}</li>" for p in text["places"]) or "<li>no data</li>"
    study = "".join(f"<p>{e(s)}</p>" for s in text["study"])
    news = []
    for a in ans.articles.itertuples(index=False):
        url = a.url if isinstance(a.url, str) and a.url.startswith(("http://", "https://")) else ""
        title = e(a.title or a.article_id)
        title = f'<a href="{e(url)}" target="_blank" rel="noopener noreferrer">{title}</a>' if url else title
        quote = f'<div class="quote">“{e(a.quote)}”</div>' if isinstance(a.quote, str) and a.quote else ""
        news.append(f"<li>{e(str(a.outlet or 'news'))}, {pd.Timestamp(a.published):%b} "
                    f"{pd.Timestamp(a.published).day} · concern {e(str(a.concern))}<br>{title}{quote}</li>")
    if not news:
        news = [f"<li class='muted'>{e(n)}</li>" for n in text["news"]]
    notes = "".join(f"<li>{e(n)}</li>" for n in text["notes"])
    return f"""
<style>
 .spark-panel {{position:fixed;top:12px;left:56px;z-index:9999;width:min(410px, 42vw);max-height:calc(100% - 40px);
   overflow-y:auto;background:rgba(255,255,255,.97);padding:12px 15px 14px;border-radius:8px;
   box-shadow:0 1px 6px rgba(0,0,0,.25);font:13px/1.4 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#0b0b0b}}
 .spark-panel .q {{font-size:16px;font-weight:600}}
 .spark-panel .chips span {{display:inline-block;background:#e9f1fb;color:#184f95;border-radius:999px;
   padding:1px 8px;margin:6px 4px 0 0;font-size:11.5px}}
 .spark-panel h4 {{margin:12px 0 4px;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#52514e}}
 .spark-panel ol, .spark-panel ul {{margin:0;padding-left:18px}} .spark-panel li {{margin:2px 0}}
 .spark-panel p {{margin:4px 0}} .spark-panel .muted, .spark-panel .notes {{color:#52514e}}
 .spark-panel .notes {{font-size:12px;margin-top:6px;padding-left:16px}}
 .spark-panel .quote {{color:#52514e;font-style:italic;margin-top:2px}}
 .spark-panel .legend {{display:flex;margin-top:4px;font-size:10.5px;color:#52514e;position:relative;
   padding-bottom:{14 if ticks else 0}px}}
 .spark-panel .legend .tick {{position:absolute;top:13px;transform:translateX(-50%);white-space:nowrap}}
 .spark-panel .legend .cell {{flex:1;text-align:center}}
 .spark-panel .legend .cell div {{height:11px;margin:0 1px 2px}}
 .spark-panel .key {{display:flex;gap:14px;align-items:center;margin-top:6px;color:#52514e;font-size:12px}}
 .spark-credit {{position:fixed;bottom:4px;left:50%;transform:translateX(-50%);z-index:9999;
   background:rgba(255,255,255,.85);padding:3px 8px;border-radius:4px;
   font:11px -apple-system,Helvetica,Arial,sans-serif;color:#52514e;white-space:nowrap}}
</style>
<div class="spark-panel">
  <div class="q">“{e(q.question or 'latest week')}”</div>
  <div class="chips">{''.join(f'<span>{e(c)}</span>' for c in chips)}</div>
  {f'<ul class="notes">{notes}</ul>' if notes else ''}
  <p style="margin-top:9px">{e(text['headline'])}</p>
  <div class="muted" style="margin-top:8px;font-size:12px">{e(unit)}</div>
  <div class="legend">{legend}</div>
  <div class="key"><span>{swatch} {outline}</span>
    <span><svg width="22" height="12"><rect x="1" y="1" width="20" height="10" fill="none" stroke="{STUDY}"
      stroke-width="2.5"/></svg> study county</span></div>
  <h4>{e(text['lead'])}</h4><ol>{places}</ol>
  {f'<h4>Study counties</h4>{study}' if study else ''}
  {f'<h4>Local news public that week</h4><ul>{"".join(news)}</ul>' if ans.study else ''}
</div>
<div class="spark-credit">Data: {e(ans.source or 'CDC NSSP')} · Boundaries: US Census 2021 ·
  Clusters: Getis-Ord Gi* / local Moran's I, Queen contiguity, permutation p, Benjamini-Hochberg FDR</div>"""


def render(ans: Answer, gaz: Gazetteer) -> folium.Map:
    """The map for one answer; ``.save(path)`` writes a self-contained HTML file."""
    q = ans.query
    frame = ans.frame
    kind = q.place.split(":")[0]
    regional = kind in ("metro", "county")
    # Simplify to what the view can show (0.02 deg ~ 2 km nationally): the
    # national file drops from ~3.6 MB to ~2 MB with no visible change.
    tolerance = {"us": 0.02, "region": 0.01, "state": 0.004}.get(kind, 0.002)
    frame = frame.assign(geometry=frame.geometry.simplify(tolerance))
    fills, cells, unit, ticks = _colors(frame, q.measure)

    fmap = folium.Map(location=ans.center, zoom_start=ans.zoom, tiles=None, control_scale=True,
                      prefer_canvas=True, zoom_snap=0.1, zoom_delta=0.5)
    if regional:
        folium.TileLayer(ESRI_GRAY, attr="Tiles &copy; Esri", name="Basemap", control=False).add_to(fmap)
    fmap.get_root().header.add_child(folium.Element(
        "<style>.leaflet-container{background:#f7f6f3 !important}</style>"))
    folium.GeoJson(
        _geojson(frame, fills, 4 if regional else 3), name="Counties",
        style_function=lambda f: {"fillColor": f["properties"]["fill"], "fillOpacity": 0.88,
                                  "color": "#ffffff", "weight": 0.4},
        highlight_function=lambda _f: {"weight": 2, "color": "#333"},
        tooltip=folium.GeoJsonTooltip(
            fields=["county", "value", "prev", "change", "trend", "cluster"],
            aliases=["County", "This week", "Week before", "Change", "CDC trend", "Cluster"], sticky=True),
    ).add_to(fmap)

    states = frame[["state", "geometry"]].dissolve("state")
    folium.GeoJson(gpd.GeoDataFrame(geometry=states.boundary.values, crs=frame.crs).__geo_interface__,
                   name="State borders", interactive=False, control=False,
                   style_function=lambda _f: {"color": "#ffffff", "weight": 1.4, "opacity": 0.9}).add_to(fmap)

    cls = frame["cluster"].astype(str)
    if q.measure == "level":
        layers = (_outline(frame, cls.str.startswith("Hot"), "Gi* hot spots", HOT, 2.5),
                  _outline(frame, cls.str.startswith("Cold"), "Gi* cold spots", COLD, 1.5, "4 4", show=False))
    else:
        kind = "High-High" if q.direction == "up" else "Low-Low"
        word = "rising" if q.direction == "up" else "falling"
        layers = (_outline(frame, cls == kind, f"Clusters of {word} counties", INK, 2.2, "5 4"),)
    study = frame["fips"].isin([f for m in METROS.values() for f in m.counties])
    for layer in (*layers, _outline(frame, study, "Study counties", STUDY, 2.5)):
        if layer is not None:
            layer.add_to(fmap)

    labels = folium.FeatureGroup(name="Labels")
    for metro in METROS.values():
        rows = frame[frame["fips"].isin(metro.counties)]
        if len(rows):
            folium.Marker((rows["lat"].mean(), rows["lon"].max() + 0.2), icon=folium.DivIcon(
                icon_size=(140, 18), icon_anchor=(0, 9),
                html=f'<div style="font:600 12px -apple-system,Helvetica,Arial;color:{INK};'
                     f'text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff;white-space:nowrap">'
                     f'{html.escape(metro.name)}</div>')).add_to(labels)
    if regional:
        for row in frame[~study].itertuples():
            folium.Marker((row.lat, row.lon), icon=folium.DivIcon(
                icon_size=(110, 14), icon_anchor=(55, 7),
                html=f'<div style="font:10px -apple-system,Helvetica,Arial;text-align:center;color:#333;'
                     f'text-shadow:0 0 2px #fff,0 0 2px #fff">'
                     f'{html.escape(row.name.removesuffix(" County"))}</div>')).add_to(labels)
    labels.add_to(fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(_panel(ans, gaz, cells, unit, ticks)))
    # Fit every drawn county in the space right of the panel, measuring the
    # panel rather than assuming its width, and refit when the window resizes.
    # Fractional zoom (zoom_snap above) lets the fit fill that space instead of
    # jumping a whole zoom level (2x) between window sizes.
    xmin, ymin, xmax, ymax = frame.total_bounds
    fmap.get_root().script.add_child(folium.Element(f"""
document.addEventListener("DOMContentLoaded", function () {{
  var map = {fmap.get_name()}, box = [[{ymin:.4f}, {xmin:.4f}], [{ymax:.4f}, {xmax:.4f}]];
  function fit() {{
    var panel = document.querySelector(".spark-panel"), left = 12;
    if (panel && panel.offsetWidth < 0.6 * window.innerWidth) left = panel.offsetLeft + panel.offsetWidth + 12;
    map.invalidateSize();
    map.fitBounds(box, {{paddingTopLeft: [left, 12], paddingBottomRight: [12, 24], animate: false}});
  }}
  var timer;
  window.addEventListener("resize", function () {{ clearTimeout(timer); timer = setTimeout(fit, 120); }});
  fit();
}});"""))
    return fmap
